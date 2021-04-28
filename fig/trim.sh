#!/bin/bash
cd "$(dirname "$0")"
mogrify -trim -path trimmed/ *.png
# see: https://imagemagick.org/discourse-server/viewtopic.php?t=17490
#convert *.png -trim -set filename:f %t_trimmed.%e +adjoin %[filename:f]
cd -
