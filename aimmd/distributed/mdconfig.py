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
import abc
import shlex
import shutil
import logging
import collections


logger = logging.getLogger(__name__)


class FlagChangeList(collections.abc.MutableSequence):
    """A list that knows if it has been changed after initializing."""

    def __init__(self, data):
        if not isinstance(data, list):
            raise TypeError("FlagChangeList must be initialized with a list.")
        self._data = data
        self._changed = False

    @property
    def changed(self):
        return self._changed

    def __repr__(self):
        return self._data.__repr__()

    def __getitem__(self, key):
        return self._data.__getitem__(key)

    def __len__(self):
        return self._data.__len__()

    def __setitem__(self, key, value):
        self._data.__setitem__(key, value)
        self._changed = True

    def __delitem__(self, key):
        self._data.__delitem__(key)
        self._changed = True

    def insert(self, key, value):
        self._data.insert(key, value)
        self._changed = True


class TypedFlagChangeList(FlagChangeList):
    """A `FlagChangeList` with an ensured type for individual list items."""

    def __init__(self, data, dtype):
        self._dtype = dtype  # set first to use in _convert_type method
        if getattr(data, '__len__', None) is None:
            # convienience for singular options,
            # if it has no len attribute we assume it is the only item
            data = [data]
        typed_data = [self._convert_type(v, key=i) for i, v in enumerate(data)]
        super().__init__(data=typed_data)

    def _convert_type(self, value, key=None):
        # here we ignore key, but passing it should in principal make it
        # possible to use different dtypes for different indices
        return self._dtype(value)

    def __setitem__(self, key, value):
        typed_value = self._convert_type(value, key=key)
        self._data.__setitem__(key, typed_value)
        self._changed = True

    def insert(self, key, value):
        typed_value = self._convert_type(value, key=key)
        self._data.insert(key, typed_value)
        self._changed = True


# NOTE: only to define the interface
class MDConfig(collections.abc.MutableMapping):
    @abc.abstractmethod
    def parse(self):
        # should read original file and populate self with key, value pairs
        raise NotImplementedError

    @abc.abstractmethod
    def write(self, outfile):
        # write out current config stored in self to outfile
        raise NotImplementedError


class LineBasedMDConfig(MDConfig):
    # abstract base class for line based parsing and writing,
    # subclasses must implement `_parse_line()` method and should set the
    # appropriate separator characters for their line format
    # We assume that every line/option can be parsed and written on its own!
    # We assume the order of the options in the written file is not relevant!
    # We represent every line/option with a key (str), list of values pair
    # values can have a specific type (e.g. int or float) or default to str.
    # NOTE: Initially written for gmx, but we already had e.g. namd in mind and
    # tried to make this as general as possible

    # these are the gmx mdp options but should be fairly general
    # (i.e. work at least for namd?)
    _KEY_VALUE_SEPARATOR = " = "
    _INTER_VALUE_CHAR = " "
    # NOTE on typing
    # use these to specify config parameters that are of type int or float
    # parsed lines with dict key matching will then be converted
    # any lines not matching will be left in their default str type
    _FLOAT_PARAMS = []  # can have multiple values per config option
    _FLOAT_SINGLETON_PARAMS = []  # must have one value per config option
    _INT_PARAMS = []
    _INT_SINGLETON_PARAMS = []
    # NOTE on SPECIAL_PARAM_DISPATCH
    # can be used to set custom type convert functions on a per parameter basis
    # the key must match the key in the dict for in the parsed line,
    # the value must be a function taking the corresponding (parsed) line and
    # which must return a FlagChangeList or subclass thereof
    # this function will also be called with the new list of value(s) when the
    # option is changed, i.e. it must also be able to check and cast a list of
    # new values into the expected FlagChangeList format
    # [note that it is probably easiest to subclass TypedFlagChangeList and
    #  overwrite only the '_check_type()' method]
    _SPECIAL_PARAM_DISPATCH = {}

    def __init__(self, original_file):
        """
        original_file - path to original config file (absolute or relative)
        """
        self._config = {}
        self._changed = False
        self._type_dispatch = self._construct_type_dispatch()
        # property to set/check file and parse to config dictionary all in one
        self.original_file = original_file

    def _construct_type_dispatch(self):
        def convert_len1_list_or_singleton(val, dtype):
            # helper func that accepts len1 lists (as expected from `_parse_line`)
            # but that also accepts single values and converts them to given type
            # (which is what we expects the users to do when they set singleton vals)
            if getattr(val, '__len__', None) is None:
                return dtype(val)
            else:
                return dtype(val[0])

        # construct type conversion dispatch
        type_dispatch = collections.defaultdict(
                                # looks a bit strange, but the factory func
                                # is called to produce the default value, i.e.
                                # we need a func that returns our default func
                                lambda:
                                lambda l: TypedFlagChangeList(data=l,
                                                              dtype=str)
                                                      )
        type_dispatch.update({param: lambda l: TypedFlagChangeList(
                                                                   data=l,
                                                                   dtype=float
                                                                   )
                              for param in self._FLOAT_PARAMS})
        type_dispatch.update({param: lambda v: convert_len1_list_or_singleton(
                                                                    val=v,
                                                                    dtype=float,
                                                                              )
                              for param in self._FLOAT_SINGLETON_PARAMS})
        type_dispatch.update({param: lambda l: TypedFlagChangeList(
                                                                   data=l,
                                                                   dtype=int,
                                                                   )
                              for param in self._INT_PARAMS})
        type_dispatch.update({param: lambda v: convert_len1_list_or_singleton(
                                                                    val=v,
                                                                    dtype=int,
                                                                              )
                              for param in self._INT_SINGLETON_PARAMS})
        type_dispatch.update(self._SPECIAL_PARAM_DISPATCH)
        return type_dispatch

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_type_dispatch"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._type_dispatch = self._construct_type_dispatch()

    @abc.abstractmethod
    def _parse_line(self, line):
        # NOTE: this is the only function needed to complete the class,
        #       the rest of this metaclass assumes the following for this func:
        # it must parse a single line and return the key, list of value(s) pair
        # as a dict with one item, e.g. {key: list of value(s)}
        # if the line is parsed as comment the dict must be empty, e.g. {}
        # if the option/key is present but without value the list must be empty
        # e.g. {key: []}
        raise NotImplementedError

    def __getitem__(self, key):
        return self._config[key]

    def __setitem__(self, key, value):
        typed_value = self._type_dispatch[key](value)
        self._config[key] = typed_value
        self._changed = True

    def __delitem__(self, key):
        self._config.__delitem__(key)
        self._changed = True

    def __iter__(self):
        return self._config.__iter__()

    def __len__(self):
        return self._config.__len__()

    def __repr__(self):
        return self._config.__repr__()

    @property
    def original_file(self):
        return self._original_file

    @original_file.setter
    def original_file(self, value):
        # NOTE: (re)setting the file also replaces the current config with
        #       what we parse from that file
        value = os.path.abspath(value)
        if not os.path.isfile(value):
            raise ValueError(f"Can not acces the file {value}")
        self._original_file = value
        self.parse()

    @property
    def changed(self):
        """Indicate if the current configuration differs from original_file."""
        # NOTE: we default to False, i.e. we expect that anything that
        #       does not have a self.changed attribute is not a container
        #       and we (the dictionary) would know that it changed
        single_vals_changed = [getattr(v, "changed", False)
                               for v in self._config.values()]
        return self._changed or any(single_vals_changed)

    def parse(self):
        """Parse the current original_file to update own state."""
        with open(self.original_file, "r") as f:
            file_content = f.read()
        parsed = {}
        # NOTE: we can split at '\n' on all platforms since py replaces
        #       all newline chars with '\n',
        #       i.e. python takes care of the differnt platforms for us :)
        for line in file_content.split("\n"):
            line_parsed = self._parse_line(line)
            # check for duplicate options, we warn but take the last one
            for key in line_parsed:
                try:
                    # check if we already have a value for that option
                    _ = parsed[key]
                except KeyError:
                    # as it should be
                    pass
                else:
                    # warn because we will only keep the last occurenc of key
                    logger.warning("Parsed duplicate configuration option "
                                   + f"({key}). Last values encountered take "
                                   + "precedence.")
            parsed.update(line_parsed)
        # convert the known types
        self._config = {key: self._type_dispatch[key](value)
                        for key, value in parsed.items()}
        self._changed = False

    def write(self, outfile, overwrite=False):
        """
        Write current configuration to outfile.

        outfile - path to outfile (relative or absolute)
        overwrite - bool (default=False), if True overwrite existing files,
                    if False and the file exists raise an error
        """
        outfile = os.path.abspath(outfile)
        if os.path.exists(outfile) and not overwrite:
            raise ValueError(f"overwrite=False and file exists ({outfile}).")
        if not self.changed:
            # just copy the original
            shutil.copy2(src=self.original_file, dst=outfile)
        else:
            # construct content for new file
            lines = []
            for key, value in self._config.items():
                line = f"{key}{self._KEY_VALUE_SEPARATOR}"
                try:
                    if len(value) > 0:
                        line += self._INTER_VALUE_CHAR.join(str(v) for v in value)
                except TypeError:
                    # (probably) not a Sequence/Iterable,
                    # i.e. one of the singleton options
                    line += f"{value}"
                lines += [line]
            # concatenate lines and write out at once
            out_str = "\n".join(lines)
            with open(outfile, "w") as f:
                f.write(out_str)


class MDP(LineBasedMDConfig):
    """
    Read, parse, (alter) and write gromacs .mdp files.

    Make all options set in a given mdp file available via a dictionary of
    option, list of values pairs. Includes automatic types for known options
    and keeps track if any options have been changed from the original.

    Notable methods:
    write - write the current (modified) configuration state to a given file
    parse - read the current original_file and update own state with it
    """

    _KEY_VALUE_SEPARATOR = " = "
    _INTER_VALUE_CHAR = " "
    # MDP param types, sorted into groups/by headings as in the gromacs manual
    # https://manual.gromacs.org/documentation/5.1/user-guide/mdp-options.html
    _FLOAT_PARAMS = []
    _FLOAT_SINGLETON_PARAMS = []
    _INT_PARAMS = []
    _INT_SINGLETON_PARAMS = []
    # Run control
    _FLOAT_SINGLETON_PARAMS += ["tinit", "dt"]
    _INT_SINGLETON_PARAMS += ["nsteps", "init-step", "simulation-part",
                              "nstcomm"]
    # Langevin dynamics
    _FLOAT_SINGLETON_PARAMS += ["bd-fric"]
    _INT_SINGLETON_PARAMS += ["ld-seed"]
    # Energy minimization
    _FLOAT_SINGLETON_PARAMS += ["emtol", "emstep"]
    _INT_SINGLETON_PARAMS += ["nstcgsteep", "nbfgscorr"]
    # Shell Molecular Dynamics
    _FLOAT_SINGLETON_PARAMS += ["fcstep"]
    _INT_SINGLETON_PARAMS += ["niter"]
    # Test particle insertion
    _FLOAT_SINGLETON_PARAMS += ["rtpi"]
    # Output control
    _FLOAT_SINGLETON_PARAMS += ["compressed-x-precision"]
    _INT_SINGLETON_PARAMS += ["nstxout", "nstvout", "nstfout", "nstlog",
                              "nstcalcenergy", "nstenergy", "nstxout-compressed"]
    # Neighbor searching
    # NOTE: 'rlistlong' and 'nstcalclr' are used with group cutoff scheme,
    #       i.e. deprecated since GMX version 5.0
    _FLOAT_SINGLETON_PARAMS += ["verlet-buffer-tolerance", "rlist", "rlistlong"]
    _INT_SINGLETON_PARAMS += ["nstlist", "nstcalclr"]
    # Electrostatics
    _FLOAT_SINGLETON_PARAMS += ["rcoulomb-switch", "rcoulomb", "epsilon-r",
                                "epsilon-rf"]
    # Van der Waals
    _FLOAT_SINGLETON_PARAMS += ["rvdw-switch", "rvdw"]
    # Ewald
    _FLOAT_SINGLETON_PARAMS += ["fourierspacing", "ewald-rtol",
                                "ewald-rtol-lj"]
    _INT_SINGLETON_PARAMS += ["fourier-nx", "fourier-ny", "fourier-nz",
                              "pme-order"]
    # Temperature coupling
    _FLOAT_PARAMS += ["tau-t", "ref-t"]
    _INT_SINGLETON_PARAMS += ["nsttcouple", "nh-chain-length"]
    # Pressure coupling
    _FLOAT_SINGLETON_PARAMS += ["tau-p"]
    _FLOAT_PARAMS += ["compressibility", "ref-p"]
    _INT_SINGLETON_PARAMS += ["nstpcouple"]
    # Simulated annealing
    _FLOAT_PARAMS += ["annealing-time", "annealing-temp"]
    _INT_PARAMS += ["annealing-npoints"]
    # Velocity generation
    _FLOAT_SINGLETON_PARAMS += ["gen-temp"]
    _INT_SINGLETON_PARAMS += ["gen-seed"]
    # Bonds
    _FLOAT_SINGLETON_PARAMS += ["shake-tol", "lincs-warnangle"]
    _INT_SINGLETON_PARAMS += ["lincs-order", "lincs-iter"]

    def _parse_line(self, line):
        parser = shlex.shlex(line, posix=True)
        parser.commenters = ";"
        parser.wordchars += "-./"  # ./ to not split floats and file paths
        tokens = list(parser)
        # gromacs mdp can have more than one token/value to the RHS of the '='
        if len(tokens) == 0:
            # (probably) a comment line
            logger.debug(f"mdp line parsed as comment: {line}")
            return {}
        elif len(tokens) >= 3 and tokens[1] == "=":
            # lines with content: make sure we correctly parsed the '='
            # always return a list for values
            return {self._key_char_replace(tokens[0]): tokens[2:]}
        elif len(tokens) == 2 and tokens[1] == "=":
            # lines with empty options, e.g. 'define = '
            return {self._key_char_replace(tokens[0]): []}
        else:
            # no idea what happend here...best to let the user have a look :)
            raise ValueError(f"Could not parse the following mdp line: {line}")

    def _key_char_replace(self, key):
        # make it possible to use CHARMM-GUI generated mdp-files, because
        # CHARMM-GUI uses "_" instead of "-" in the option names,
        # which seems to be an undocumented gromacs feature,
        # i.e. gromacs reads these mdp-files without complaints :)
        # we will however stick with "-" all the time to make sure every option
        # exists only once, i.e. we convert all keys to use "-" instead of "_"
        return key.replace("_", "-")

    def __getitem__(self, key):
        return super().__getitem__(self._key_char_replace(key))

    def __setitem__(self, key, value):
        return super().__setitem__(self._key_char_replace(key), value)

    def __delitem__(self, key):
        return super().__delitem__(self._key_char_replace(key))
