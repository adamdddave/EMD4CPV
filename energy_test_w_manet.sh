#!/bin/bash
echo "A simple wrapper which will run the MANET code on two locations. Consider setting in this script the variables LOCATION_OF_DATA and LOCATION_OF_OUTPUT"
echo "./Manet_noROOT can be found from the get_manet_svn.sh script"
LOCATION_OF_DATA=/my/path/to/data/
LOCATION_OF_OUTPUT=/path/to/my/output/

./Manet_noROOT -f1 $LOCATION_OF_DATA/X_m2_temp.txt -f2 LOCATION_OF_DATA/Xbar_m2_temp.txt -s 0.4 -p 0

mv Ts.txt $LOCATION_OF_OUTPUT/T_w_manet_CPV.txt
