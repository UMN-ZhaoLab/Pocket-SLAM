#!/bin/bash

code_path='' # path to lsg slam code
config_path=$code_path'/configs/euroc/lsgslam.py'

scene_names=(
"MH_01_easy" 
"MH_02_easy" 
"MH_03_medium" 
"MH_04_difficult" 
"MH_05_difficult" 
"V1_01_easy" 
"V1_02_medium" 
"V1_03_difficult" 
"V2_01_easy" 
"V2_02_medium" 
"V2_03_difficult"
)
start_idx_array=(900 780 350 368 400 0 0 0 0 0 0)
end_idx_array=(3630 2990 2600 1970 2220 2870 1670 2090 2200 2300 1890)

image_width=752
image_height=480

# for((j=3;j<=4;j+=1));
for j in 8
do 
    scene_name=${scene_names[j]}
    start=${start_idx_array[j]}
    end=${end_idx_array[j]}
    step=200
    # step=${end-start}

    echo $scene_name
    echo $start
    echo $end
    echo $step
    
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

        n=`grep -n "image_width = " $config_path | awk -F':' '{print $1}'` 
        sed -i "$[ n ]c image_width = $image_width" $config_path

        n=`grep -n "image_height = " $config_path | awk -F':' '{print $1}'` 
        sed -i "$[ n ]c image_height = $image_height" $config_path

        n=`grep -n "start_idx = " $config_path | awk -F':' '{print $1}'` 
        sed -i "$[ n ]c start_idx = $start_idx" $config_path

        n=`grep -n "end_idx = " $config_path | awk -F':' '{print $1}'` 
        sed -i "$[ n ]c end_idx = $end_idx" $config_path

        cd $code_path
        python3 scripts/splatam.py $config_path

    done

    n=`grep -n "scene_name = " $config_path | awk -F':' '{print $1}'` 
    sed -i "$[ n ]c scene_name = '$scene_name'" $config_path

    n=`grep -n "image_width = " $config_path | awk -F':' '{print $1}'` 
    sed -i "$[ n ]c image_width = $image_width" $config_path

    n=`grep -n "image_height = " $config_path | awk -F':' '{print $1}'` 
    sed -i "$[ n ]c image_height = $image_height" $config_path

    n=`grep -n "start_idx = " $config_path | awk -F':' '{print $1}'` 
    sed -i "$[ n ]c start_idx = $start" $config_path

    n=`grep -n "end_idx = " $config_path | awk -F':' '{print $1}'` 
    sed -i "$[ n ]c end_idx = $end" $config_path

    cd $code_path
    python3 scripts/loop_closure.py $config_path

done