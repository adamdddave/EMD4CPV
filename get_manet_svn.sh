#!/bin/bash
#get_manet_svn.sh

echo "A simple wrapper script to download the MANET code from SVN. Follows instructions from"
echo "Please feel free to do whatever else you need from https://manet.hepforge.org/"
svn checkout svn+ssh://vcs@phab.hepforge.org:22/source/manetsvn manetsvn