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
import shlex
import asyncio
import subprocess


class SlurmProcess:
    """
    Generic wrapper around SLURM submissions.

    Imitates the interface of asyncio.subprocess.Process
    """

    # we can not simply wait for the subprocess, since slurm exits directly
    # so we will sleep for this long between checks if slurm-job completed
    sleep_time = 30  # TODO: heuristic? dynamically adapt?
    # NOTE: no options to set/pass extra_args for sbatch and sacct:
    #       I think all sbatch options can also be set via SBATCH directives?!
    #       and sacct options would probably only mess up our parsing... ;)
    sacct_executable = "sacct"
    sbatch_executable = "sbatch"
    scancel_executable = "scancel"
    # rudimentary map for slurm state codes to int return codes for poll
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
        self.sbatch_script = os.path.abspath(sbatch_script)
        self.workdir = os.path.abspath(workdir)
        self._jobid = None

    async def submit(self):
        sbatch_cmd = f"{self.sbatch_executable} --parsable {self.sbatch_script}"
        sbatch_proc = await asyncio.subprocess.create_subprocess_exec(
                                                *shlex.split(sbatch_cmd),
                                                stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.PIPE,
                                                cwd=self.workdir,
                                                                      )
        stdout, stderr = await sbatch_proc.communicate()
        sbatch_return = stdout.decode()
        # only jobid (and possibly clustername) returned, semikolon to separate
        jobid = sbatch_return.split(";")[0].strip()
        self._jobid = jobid

    @property
    def slurm_jobid(self):
        return self._jobid

    @property
    def slurm_job_state(self):
        if self._jobid is None:
            return None
        sacct_cmd = f"{self.sacct_executable} --noheader"
        # query only for the specific job we are running
        sacct_cmd += f" -j {self._jobid}"
        sacct_cmd += " -o jobid,state,exitcode --parsable2"  # separate with |
        sacct_out = subprocess.check_output(shlex.split(sacct_cmd), text=True)
        # sacct returns one line per substep, we only care for the whole job
        # which should be the first line but we check explictly for jobid
        # (the substeps have .$NUM suffixes)
        for line in sacct_out.split("\n"):
            splits = line.split("|")
            if len(splits) == 3:
                jobid, state, exitcode = splits
                if jobid.strip() == self._jobid:
                    # TODO: parse and return the exitcode too?
                    return state
        # if we get here something probably went wrong checking for the job
        # the 'PENDING' will make us check again in a bit
        # (TODO: is this actually what we want?)
        return "PENDING"

    @property
    def returncode(self):
        slurm_state = self.slurm_job_state
        if slurm_state is None:
            return None
        for key, val in self.slurm_state_to_exitcode.items():
            if key in slurm_state:
                # this also recognizes `CANCELLED by ...` as CANCELLED
                return val
        # we should never finish the loop, it means we miss a slurm job state
        raise RuntimeError("Could not find a matching exitcode for slurm state"
                           + f": {slurm_state}")

    async def wait(self):
        while self.returncode is None:
            await asyncio.sleep(self.sleep_time)
        return self.returncode

    async def send_signal(signal):
        # TODO: write this! (if we actually need it?)
        raise NotImplementedError

    def terminate(self):
        if self._jobid is not None:
            scancel_cmd = f"{self.scancel_executable} {self._jobid}"
            # TODO: parse/check output?!
            scancel_out = subprocess.check_output(shlex.split(scancel_cmd),
                                                  text=True)

    def kill(self):
        self.terminate()
