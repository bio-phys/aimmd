"""
This file is part of AIMMD.

AIMMD is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

AIMMD is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with AIMMD. If not, see <https://www.gnu.org/licenses/>.
"""
import os
import time
import shlex
import asyncio
import subprocess
import collections
import logging


from . import _SEMAPHORES
from .utils import ensure_executable_available


logger = logging.getLogger(__name__)


def list_all_nodes(sinfo_executable="sinfo"):
    # format option '%n' is a list of node hostnames
    sinfo_cmd = f"{sinfo_executable} --noheader --format='%n'"
    try:
        sinfo_out = subprocess.check_output(shlex.split(sinfo_cmd), text=True)
    except FileNotFoundError:  # raise when there is no sinfo command
        logger.warning("sinfo command not available. Initializing list of all "
                       + "nodes in cluster as empty.")
        node_list = []
    else:
        node_list = sinfo_out.split("\n")
        # sinfo_out is terminated by '\n' so our last entry is the empty string
        node_list = node_list[:-1]
    return node_list


_SLURM_CLUSTER_INFO = {"nodes": {"all": list_all_nodes(),
                                 "suspected_broken": collections.Counter(),
                                 "broken": [],
                                 },
                       }


# TODO: use slurm_job_state to decide if we even supect the node is broken?!
#       (currently also stuff like out of memory counts into the 'node failures')
# TODO: make max_fail an attribute of SlurmProcess?
async def _handle_suspected_broken_nodes(listofnodes, slurm_job_state,
                                         max_fail=3):
    # max_fail is the maximum number of jobs we allow to fail on one node
    # before declaring it broken
    global _SLURM_CLUSTER_INFO
    async with _SEMAPHORES["SLURM_CLUSTER_INFO"]:
        logger.debug(f"Adding nodes {listofnodes} to suspected broken nodes.")
        for node in listofnodes:
            _SLURM_CLUSTER_INFO["nodes"]["suspected_broken"][node] += 1
            if _SLURM_CLUSTER_INFO["nodes"]["suspected_broken"][node] >= max_fail:
                # declare it broken
                logger.info(f"Adding node {node} to list of broken nodes.")
                if node not in _SLURM_CLUSTER_INFO["nodes"]["broken"]:
                    _SLURM_CLUSTER_INFO["nodes"]["broken"].append(node)
                else:
                    logger.debug(f"Node {node} already in broken node list.")
        # failsaves
        all_nodes = len(_SLURM_CLUSTER_INFO["nodes"]["all"])
        broken_nodes = len(_SLURM_CLUSTER_INFO["nodes"]["broken"])
        if broken_nodes >= all_nodes / 4:
            logger.error("We already declared 1/4 of the cluster as broken."
                         + "Houston, we might have a problem?")
            if broken_nodes >= all_nodes / 2:
                logger.error("In fact we declared 1/2 of the cluster as broken."
                             + "Houston, we have a problem!")
                if broken_nodes >= all_nodes * 0.75:
                    raise RuntimeError("Houston? Almost the whole cluster is broken?")


class SlurmSubmissionError(RuntimeError):
    """Error raised when something goes wrong submitting a SLURM job."""


class SlurmProcess:
    """
    Generic wrapper around SLURM submissions.

    Imitates the interface of `asyncio.subprocess.Process`
    """

    # we can not simply wait for the subprocess, since slurm exits directly
    # so we will sleep for this long between checks if slurm-job completed
    sleep_time = 30  # TODO: heuristic? dynamically adapt?
    min_time_between_sacct_calls = 10  # wait for at least 10 s between two sacct calls
    # NOTE: no options to set/pass extra_args for sbatch and sacct:
    #       I think all sbatch options can also be set via SBATCH directives?!
    #       and sacct options would probably only mess up our parsing... ;)
    sacct_executable = "sacct"
    sbatch_executable = "sbatch"
    scancel_executable = "scancel"
    # rudimentary map for slurm state codes to int return codes for poll
    # NOTE: these are the sacct states (they differ from the squeue states)
    #       cf. https://slurm.schedmd.com/sacct.html#lbAG
    #       and https://slurm.schedmd.com/squeue.html#lbAG
    slurm_state_to_exitcode = {
        "BOOT_FAIL": 1,  # Job terminated due to launch failure
        # Job was explicitly cancelled by the user or system administrator.
        "CANCELLED": 1,
        # Job has terminated all processes on all nodes with an exit code of
        # zero.
        "COMPLETED": 0,
        "DEADLINE": 1,  # Job terminated on deadline.
        # Job terminated with non-zero exit code or other failure condition.
        "FAILED": 1,
        # Job terminated due to failure of one or more allocated nodes.
        "NODE_FAIL": 1,
        "OUT_OF_MEMORY": 1,  # Job experienced out of memory error.
        "PENDING": None,  # Job is awaiting resource allocation.
        # NOTE: preemption means interupting a process to later restart it,
        #       i.e. None is probably the right thing to return
        "PREEMPTED": None,  # Job terminated due to preemption.
        "RUNNING": None,  # Job currently has an allocation.
        "REQUEUED": None,  # Job was requeued.
        # Job is about to change size.
        #"RESIZING" TODO: when does this happen? what should we return?
        # Sibling was removed from cluster due to other cluster starting the
        # job.
        "REVOKED": 1,
        # Job has an allocation, but execution has been suspended and CPUs have
        # been released for other jobs.
        "SUSPENDED": None,
        # Job terminated upon reaching its time limit.
        "TIMEOUT": 1,  # TODO: can this happen for jobs that finish properly?
    }

    def __init__(self, sbatch_script, workdir, **kwargs):
        # we expect sbatch_script to be a path to a file
        # make it possible to set any attribute via kwargs
        # check the type for attributes with default values
        dval = object()
        for kwarg, value in kwargs.items():
            cval = getattr(self, kwarg, dval)
            if cval is not dval:
                if isinstance(value, type(cval)):
                    # value is of same type as default so set it
                    setattr(self, kwarg, value)
                else:
                    raise TypeError(f"Setting attribute {kwarg} with "
                                    + f"mismatching type ({type(value)}). "
                                    + f" Default type is {type(cval)}."
                                    )
        # this either checks for our defaults or whatever we just set via kwargs
        ensure_executable_available(self.sacct_executable)
        ensure_executable_available(self.sbatch_executable)
        ensure_executable_available(self.scancel_executable)
        self.sbatch_script = os.path.abspath(sbatch_script)
        self.workdir = os.path.abspath(workdir)
        self._jobid = None
        self._nodes = None  # list of node hostnames this job runs on
        self._last_check_time = None  # make sure we do not call sacct too often
        # to that avail cache the last sacct return we have seen
        self._last_sacct_return = (None, None, None)

    async def submit(self):
        global _SLURM_CLUSTER_INFO
        sbatch_cmd = f"{self.sbatch_executable}"
        broken_nodes = _SLURM_CLUSTER_INFO["nodes"]["broken"]
        if len(broken_nodes) > 0:
            sbatch_cmd += f" --exclude={','.join(broken_nodes)}"
        sbatch_cmd += f" --parsable {self.sbatch_script}"
        logger.debug(f"About to execute sbatch_cmd {sbatch_cmd}.")
        # 3 file descriptors: stdin,stdout,stderr
        await _SEMAPHORES["MAX_FILES_OPEN"].acquire()
        await _SEMAPHORES["MAX_FILES_OPEN"].acquire()
        await _SEMAPHORES["MAX_FILES_OPEN"].acquire()
        try:
            sbatch_proc = await asyncio.subprocess.create_subprocess_exec(
                                                *shlex.split(sbatch_cmd),
                                                stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.PIPE,
                                                cwd=self.workdir,
                                                close_fds=True,
                                                                          )
            stdout, stderr = await sbatch_proc.communicate()
            sbatch_return = stdout.decode()
        finally:
            # and put the three back into the semaphore
            _SEMAPHORES["MAX_FILES_OPEN"].release()
            _SEMAPHORES["MAX_FILES_OPEN"].release()
            _SEMAPHORES["MAX_FILES_OPEN"].release()
        # only jobid (and possibly clustername) returned, semikolon to separate
        logger.debug(f"sbatch returned stdout: {sbatch_return}, "
                     + f"stderr: {stderr.decode()}.")
        jobid = sbatch_return.split(";")[0].strip()
        # make sure jobid is an int/ can be cast as one
        err = False
        try:
            jobid_int = int(jobid)
        except ValueError:
            # can not cast to int, so probably something went wrong submitting
            err = True
        else:
            if str(jobid_int) != jobid:
                err = True
        if err:
            raise SlurmSubmissionError("Could not submit SLURM job."
                                       + f" sbatch returned {sbatch_return}.")
        logger.info(f"Submited SLURM job with jobid {jobid}.")
        self._jobid = jobid

    @property
    def slurm_jobid(self):
        return self._jobid

    @property
    def nodes(self):
        nodes = None
        slurm_nodelist = self._last_sacct_return[2]
        if slurm_nodelist is not None:
            nodes = self._nodelist_to_listofnodes(slurm_nodelist)
        return nodes

    async def _get_sacct_jobinfo(self):
        if self._last_check_time is None:
            self._last_check_time = time.time()
        elif (time.time() - self._last_check_time
              <= self.min_time_between_sacct_calls):
            return self._last_sacct_return
        else:
            self._last_check_time = time.time()
        sacct_cmd = f"{self.sacct_executable} --noheader"
        # query only for the specific job we are running
        sacct_cmd += f" -j {self._jobid}"
        sacct_cmd += " -o jobid,state,exitcode,nodelist"
        sacct_cmd += " --parsable2"  # separate with |
        # 3 file descriptors: stdin,stdout,stderr
        await _SEMAPHORES["MAX_FILES_OPEN"].acquire()
        await _SEMAPHORES["MAX_FILES_OPEN"].acquire()
        await _SEMAPHORES["MAX_FILES_OPEN"].acquire()
        try:
            sacct_proc = await asyncio.subprocess.create_subprocess_exec(
                                                *shlex.split(sacct_cmd),
                                                stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.PIPE,
                                                cwd=self.workdir,
                                                close_fds=True,
                                                                          )
            stdout, stderr = await sacct_proc.communicate()
            sacct_return = stdout.decode()
        finally:
            # and put the three back into the semaphore
            _SEMAPHORES["MAX_FILES_OPEN"].release()
            _SEMAPHORES["MAX_FILES_OPEN"].release()
            _SEMAPHORES["MAX_FILES_OPEN"].release()
        # only jobid (and possibly clustername) returned, semikolon to separate
        logger.debug(f"sacct returned {sacct_return}.")
        # sacct returns one line per substep, we only care for the whole job
        # which should be the first line but we check explictly for jobid
        # (the substeps have .$NUM suffixes)
        for line in sacct_return.split("\n"):
            splits = line.split("|")
            if len(splits) == 4:
                jobid, state, exitcode, nodelist = splits
                if jobid.strip() == self._jobid:
                    # basic sanity check that everything went alright parsing
                    logger.debug(f"Extracted from sacct output: jobid {jobid},"
                                 + f" state {state}, exitcode {exitcode} and "
                                 + f"nodelist {nodelist}.")
                    self._last_sacct_return = (state, exitcode, nodelist)
                    return state, exitcode, nodelist
                else:
                    raise RuntimeError("Something went horribly wrong calling"
                                       + " sacct and parsing its output. We "
                                       + f"parsed jobid {jobid} but expected "
                                       + f"jobid to be {self._jobid}.")
        # if we get here something probably went wrong checking for the job
        # the 'PENDING' will make us check again in a bit
        # (TODO: is this actually what we want?)
        self._last_sacct_return = ("PENDING", None, None)
        return "PENDING", None, None

    @property
    def slurm_job_state(self):
        return self._last_sacct_return[0]

    @property
    def returncode(self):
        if self._jobid is None:
            return None
        if self.slurm_job_state is None:
            return None
        for key, val in self.slurm_state_to_exitcode.items():
            if key in self.slurm_job_state:
                logger.debug(f"Parsed SLURM state {self.slurm_job_state} as {key}.")
                # this also recognizes `CANCELLED by ...` as CANCELLED
                return val
        # we should never finish the loop, it means we miss a slurm job state
        raise RuntimeError("Could not find a matching exitcode for slurm state"
                           + f": {self.slurm_job_state}")

    async def wait(self):
        while self.returncode is None:
            await asyncio.sleep(self.sleep_time)
            await self._get_sacct_jobinfo()  # update cached jobinfo
        if self.returncode != 0:
            await _handle_suspected_broken_nodes(
                                        nodelist=self.nodes,
                                        slurm_job_state=self.slurm_job_state,
                                                 )
        return self.returncode

    async def communicate(self, input=None):
        # TODO: write this (if we need it)
        #       [at least the reading from stdout, stderr should be doable
        #        if we know the slurm output files for that]
        raise NotImplementedError

    def send_signal(self, signal):
        # TODO: write this! (if we actually need it?)
        #       [should be doable via scancel, which can send signals to jobs]
        raise NotImplementedError

    def terminate(self):
        if self._jobid is not None:
            scancel_cmd = f"{self.scancel_executable} {self._jobid}"
            # TODO: parse/check output?!
            scancel_out = subprocess.check_output(shlex.split(scancel_cmd),
                                                  text=True)
            logger.debug(f"Canceled SLURM job with jobid {self.slurm_jobid}."
                         + f"scancel returned {scancel_out}.")

    def kill(self):
        self.terminate()

    def _nodelist_to_listofnodes(self, nodelist):
        # takes a NodeList as returned by SLURMs sacct
        # returns a list of single node hostnames
        # NOTE: we expect nodelist to be either a string of the form
        # $hostnameprefix$num or $hostnameprefix[$num1,$num2,...,$numN]
        # or 'None assigned'
        if "[" not in nodelist:
            # it is '$hostnameprefix$num' or 'None assigned', return it
            return [nodelist]
        else:
            # it is '$hostnameprefix[$num1,$num2,...,$numN]'
            # make the string a list of single node hostnames
            hostnameprefix, nums = nodelist.split("[")
            nums = nums.rstrip("]")
            nums = nums.split(",")
            return [f"{hostnameprefix}{num}" for num in nums]
