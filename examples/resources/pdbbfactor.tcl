
# load multi-frame pdb file, storing B factors from each frame in user.
# usage: pdbbfactor <filename>
#
# Justin Gullingsrud
# 3 September 2004
#
# modified March 2019 to work for mdtraj generated pdbs, Hendrik Jung


proc pdbbfactor { fname } {
  mol new $fname waitfor all
  set all [atomselect top all]
  set frame 0
  set in [open $fname r]
  set beta {}
  while { [gets $in line] != -1 } {
    switch -- [string range $line 0 3] {
      ENDM {
        $all frame $frame
        $all set user $beta
        set beta {}
        incr frame
      }
      ATOM -
      HETA {
        lappend beta [string range $line 61 66]
      }
    }
  }
}

