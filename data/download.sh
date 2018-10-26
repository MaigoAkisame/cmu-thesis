archives="audioset.tgz sequential.tgz dcase.tgz"
for archive in $archives; do
  wget http://islpc21.is.cs.cmu.edu/yunwang/git/cmu-thesis/data/$archive && ((tar zxf $archive && rm $archive) &)
done
while [ $(ls $archives 2>/dev/null | wc -l) -ne 0 ]; do
  echo -ne "Extracting file $(ls ${archives//.tgz/\/*} 2>/dev/null | wc -l) of 47457 ...\r"
  sleep 10;
done
echo -e "\nAll files extracted. DONE!"
