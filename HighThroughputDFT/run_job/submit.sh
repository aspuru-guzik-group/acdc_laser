workdir=$PWD
input=$1
line="$(head -n 1 $input)"

if [[ -z "$line" ]]; then
  exit 0
else

  tail -n +2 $input > tmp.inp && mv tmp.inp $input 
  name="$(echo $line | awk '{print $2}')"
  smi="$(echo $line | awk '{print $1}')"
  smi="$(echo $smi | sed 's?\\?\\\\?g')"
  echo $name $smi
  cd $workdir

  while test -d ./$name; do
    if [[ -z "$name" ]]; then
      exit 0
    fi
    echo "$name exist"
    line="$(head -n 1 $input)"
    tail -n +2 $input > tmp.inp && mv tmp.inp $input
    name="$(echo $line | awk '{print $2}')"
    smi="$(echo $line | awk '{print $1}')"
    smi="$(echo $smi | sed 's?\\?\\\\?g')"
    echo $name $smi
  done

  mkdir $name
  cd $name
  cp $HOME/madness/madness_job.sh .
  sed -i -e "s/ppppp/16/g" madness_job.sh
  sed -i -e "s/nnnnn/$name/g" madness_job.sh
  sed -i -e "s?smismismi?$smi?g" madness_job.sh #use ? as deliminator for smiles as it will not appear in smiles
  sed -i -e "s/inputinputinput/$input/g" madness_job.sh
  sbatch madness_job.sh

fi
