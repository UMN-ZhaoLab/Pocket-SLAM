#!/bin/bash

code_path='' # path to lsg slam code
config_path=$code_path'/configs/tum/lsgslam.py'

scene_names=(
"freiburg1_desk"  
"freiburg2_xyz" 
"freiburg3_long_office_household"
)
start_idx_array=(0 0 0)
end_idx_array=(595 3666 2509)

for((j=2;j<=2;j+=1));
do 
    scene_name=${scene_names[j]}
    start=${start_idx_array[j]}
    end=${end_idx_array[j]}

    step=200
    echo $scene_name
    echo $start
    echo $end
    for((i=$start;i<=$end;i+=$step));
    do 
        start_idx=$i
        let end_idx=i+step
        if [ $end_idx -ge $end ]; then
            end_idx=$end
        fi
        if [ $start_idx -eq $end_idx ]; then
            break
        fi
        echo echo "Processing $start_idx to $end_idx"

        n=`grep -n "scene_name = " $config_path | awk -F':' '{print $1}'` 
        sed -i "$[ n ]c scene_name = '$scene_name'" $config_path

        n=`grep -n "start_idx = " $config_path | awk -F':' '{print $1}'` 
        sed -i "$[ n ]c start_idx = $start_idx" $config_path

        n=`grep -n "end_idx = " $config_path | awk -F':' '{print $1}'` 
        sed -i "$[ n ]c end_idx = $end_idx" $config_path

        cd $code_path
        python3 scripts/splatam.py $config_path

    done

    n=`grep -n "scene_name = " $config_path | awk -F':' '{print $1}'` 
    sed -i "$[ n ]c scene_name = '$scene_name'" $config_path

    n=`grep -n "start_idx = " $config_path | awk -F':' '{print $1}'` 
    sed -i "$[ n ]c start_idx = $start" $config_path

    n=`grep -n "end_idx = " $config_path | awk -F':' '{print $1}'` 
    sed -i "$[ n ]c end_idx = $end" $config_path

    cd $code_path
    python3 scripts/loop_closure.py $config_path

done