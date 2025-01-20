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
import subprocess


def _get_version_from_pyproject():
    """Get version string from pyproject.toml file."""
    pyproject_toml = os.path.join(os.path.dirname(__file__),
                                  "../../pyproject.toml")
    with open(pyproject_toml) as f:
        line = f.readline()
        while line:
            if line.startswith("version ="):
                version_line = line
                break
            line = f.readline()
    version = version_line.strip().split(" = ")[1]
    version = version.replace('"', '').replace("'", "")
    return version


def _get_git_hash_and_tag():
    """Get git hash, date, and tag from git log."""
    git_hash = ""
    git_date = ""
    git_tag = ""
    p = subprocess.Popen(
            ["git", "log", "-1", "--format='%H || %as || %(describe:tags=true,match=v*)'"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.dirname(__file__),
                         )
    stdout, stderr = p.communicate()
    if p.returncode == 0:
        git_hash, git_date, git_describe = (stdout.decode("utf-8")
                                            .replace("'", "").replace('"', '')
                                            .strip().split("||"))
        git_date = git_date.strip().replace("-", "")
        git_describe = git_describe.strip()
        if "-" not in git_describe and git_describe != "":
            # git-describe returns either the git-tag or (if we are not exactly
            #  at a tag) something like
            #  $GITTAG-$NUM_COMMITS_DISTANCE-$CURRENT_COMMIT_HASH
            git_tag = git_describe[1:]  # strip of the 'v'
    return git_hash, git_date, git_tag

try:
    _version = _get_version_from_pyproject()
except FileNotFoundError:
    # pyproject.toml not found
    import importlib.metadata
    __version__ = importlib.metadata.version("aimmd")
    __git_hash__ = ""
else:
    _git_hash, _git_date, _git_tag = _get_git_hash_and_tag()
    __git_hash__ = _git_hash
    if _version == _git_tag or _git_hash == "":
        # dont append git_hash to version, if it is a version-tagged commit or if
        # git_hash is empty (happens if git is installed but we are not in a repo)
        __version__ = _version
    else:
        __version__ = _version + f"+git{_git_date}.{_git_hash[:7]}"
