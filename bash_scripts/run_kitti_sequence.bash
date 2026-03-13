#!/bin/bash

code_path='' # path to lsg slam code
config_path=$code_path'/configs/kitti/lsgslam.py'

# scene_name, start_idx, end_idx, image_width, image_height, calib_file
scene_names=(
"00,50,4540,2,1241,376,./configs/kitti/kitti00-02.yaml" 
"01,0,1100,2,1241,376,./configs/kitti/kitti00-02.yaml" 
"02,0,4660,2,1241,376,./configs/kitti/kitti00-02.yaml" 
"03,0,800,2,1242,375,./configs/kitti/kitti03.yaml" 
"04,0,270,2,1226,370,./configs/kitti/kitti04-10.yaml" 
"05,0,2760,2,1226,370,./configs/kitti/kitti04-10.yaml" 
"06,0,1100,2,1226,370,./configs/kitti/kitti04-10.yaml" 
"07,0,1100,2,1226,370,./configs/kitti/kitti04-10.yaml" 
"08,0,4070,2,1226,370,./configs/kitti/kitti04-10.yaml" 
"09,0,1590,2,1226,370,./configs/kitti/kitti04-10.yaml" 
"10,0,1200,2,1226,370,./configs/kitti/kitti04-10.yaml" 
# "11,0,920" 
# "12,0,1060" 
# "13,0,3280" 
# "14,0,630" 
# "15,0,1900" 
# "16,0,1730" 
# "17,0,490" 
# "18,0,1800" 
# "19,0,4980" 
# "20,0,830" 
# "21,0,2720" 
)

step=50

for j in 3;
# for((j=1;j<=10;j+=1));
do 
    array=(${scene_names[j]//,/ })  
    scene_name=${array[0]}
    start=${array[1]}
    end=${array[2]}
    stride=${array[3]}
    image_width=${array[4]}
    image_height=${array[5]}
    kitti_yaml=${array[6]}

    echo $scene_name
    echo $start
    echo $end
    echo $stride
    echo $image_width
    echo $image_height
    echo $calib_file

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

        n=`grep -n "kitti_yaml = " $config_path | awk -F':' '{print $1}'` 
        sed -i "$[ n ]c kitti_yaml = '$kitti_yaml'" $config_path

        n=`grep -n "image_width = " $config_path | awk -F':' '{print $1}'` 
        sed -i "$[ n ]c image_width = $image_width" $config_path

        n=`grep -n "image_height = " $config_path | awk -F':' '{print $1}'` 
        sed -i "$[ n ]c image_height = $image_height" $config_path

        n=`grep -n "start_idx = " $config_path | awk -F':' '{print $1}'` 
        sed -i "$[ n ]c start_idx = $start_idx" $config_path

        n=`grep -n "end_idx = " $config_path | awk -F':' '{print $1}'` 
        sed -i "$[ n ]c end_idx = $end_idx" $config_path

        n=`grep -n "stride = " $config_path | awk -F':' '{print $1}'` 
        sed -i "$[ n ]c stride = $stride" $config_path

        cd $code_path
        python3 scripts/splatam.py $config_path

    done

    n=`grep -n "scene_name = " $config_path | awk -F':' '{print $1}'` 
    sed -i "$[ n ]c scene_name = '$scene_name'" $config_path

    n=`grep -n "kitti_yaml = " $config_path | awk -F':' '{print $1}'` 
    sed -i "$[ n ]c kitti_yaml = '$kitti_yaml'" $config_path

    n=`grep -n "image_width = " $config_path | awk -F':' '{print $1}'` 
    sed -i "$[ n ]c image_width = $image_width" $config_path

    n=`grep -n "image_height = " $config_path | awk -F':' '{print $1}'` 
    sed -i "$[ n ]c image_height = $image_height" $config_path

    n=`grep -n "start_idx = " $config_path | awk -F':' '{print $1}'` 
    sed -i "$[ n ]c start_idx = $start" $config_path

    n=`grep -n "end_idx = " $config_path | awk -F':' '{print $1}'` 
    sed -i "$[ n ]c end_idx = $end" $config_path

    n=`grep -n "stride = " $config_path | awk -F':' '{print $1}'` 
    sed -i "$[ n ]c stride = $stride" $config_path

    cd $code_path
    python3 scripts/loop_closure.py $config_path
done
