rm -f run.sh
ARCH=500-500-500
SIZE=20000000,250000,250000,65000000
cat $PWD/config.rz.yml | sed "s/ARCH/$ARCH/g;s/SIZE/$SIZE/g" > $PWD/config.rz.$ARCH.$SIZE.yml
echo "CUDA_VISIBLE_DEVICES=0 nohup python3 /home/apd10/ExtremeClassificationRz/train.py --tmpdir /home/apd10/ExtremeClassificationRz/runs/exp1/  --config $PWD/config.rz.$ARCH.$SIZE.yml &" >> run.sh
SIZE=2000000,25000,25000,6500000
cat $PWD/config.rz.yml | sed "s/ARCH/$ARCH/g;s/SIZE/$SIZE/g" > $PWD/config.rz.$ARCH.$SIZE.yml
echo "CUDA_VISIBLE_DEVICES=0 nohup python3 /home/apd10/ExtremeClassificationRz/train.py --tmpdir /home/apd10/ExtremeClassificationRz/runs/exp1/  --config $PWD/config.rz.$ARCH.$SIZE.yml &" >> run.sh
SIZE=200000,25000,25000,650000
cat $PWD/config.rz.yml | sed "s/ARCH/$ARCH/g;s/SIZE/$SIZE/g" > $PWD/config.rz.$ARCH.$SIZE.yml
echo "CUDA_VISIBLE_DEVICES=0 nohup python3 /home/apd10/ExtremeClassificationRz/train.py --tmpdir /home/apd10/ExtremeClassificationRz/runs/exp1/  --config $PWD/config.rz.$ARCH.$SIZE.yml &" >> run.sh
SIZE=20000,10000,10000,65000
cat $PWD/config.rz.yml | sed "s/ARCH/$ARCH/g;s/SIZE/$SIZE/g" > $PWD/config.rz.$ARCH.$SIZE.yml
echo "CUDA_VISIBLE_DEVICES=0 nohup python3 /home/apd10/ExtremeClassificationRz/train.py --tmpdir /home/apd10/ExtremeClassificationRz/runs/exp1/  --config $PWD/config.rz.$ARCH.$SIZE.yml &" >> run.sh

ARCH=1000-1000-1000
SIZE=20000000,250000,250000,65000000
cat $PWD/config.rz.yml | sed "s/ARCH/$ARCH/g;s/SIZE/$SIZE/g" > $PWD/config.rz.$ARCH.$SIZE.yml
echo "CUDA_VISIBLE_DEVICES=0 nohup python3 /home/apd10/ExtremeClassificationRz/train.py --tmpdir /home/apd10/ExtremeClassificationRz/runs/exp1/  --config $PWD/config.rz.$ARCH.$SIZE.yml &" >> run.sh
SIZE=2000000,25000,25000,6500000
cat $PWD/config.rz.yml | sed "s/ARCH/$ARCH/g;s/SIZE/$SIZE/g" > $PWD/config.rz.$ARCH.$SIZE.yml
echo "CUDA_VISIBLE_DEVICES=0 nohup python3 /home/apd10/ExtremeClassificationRz/train.py --tmpdir /home/apd10/ExtremeClassificationRz/runs/exp1/  --config $PWD/config.rz.$ARCH.$SIZE.yml &" >> run.sh
SIZE=200000,25000,25000,650000
cat $PWD/config.rz.yml | sed "s/ARCH/$ARCH/g;s/SIZE/$SIZE/g" > $PWD/config.rz.$ARCH.$SIZE.yml
echo "CUDA_VISIBLE_DEVICES=0 nohup python3 /home/apd10/ExtremeClassificationRz/train.py --tmpdir /home/apd10/ExtremeClassificationRz/runs/exp1/  --config $PWD/config.rz.$ARCH.$SIZE.yml &" >> run.sh
SIZE=20000,10000,10000,65000
cat $PWD/config.rz.yml | sed "s/ARCH/$ARCH/g;s/SIZE/$SIZE/g" > $PWD/config.rz.$ARCH.$SIZE.yml
echo "CUDA_VISIBLE_DEVICES=0 nohup python3 /home/apd10/ExtremeClassificationRz/train.py --tmpdir /home/apd10/ExtremeClassificationRz/runs/exp1/  --config $PWD/config.rz.$ARCH.$SIZE.yml &" >> run.sh

ARCH=5000-5000-1000
SIZE=20000000,250000,250000,65000000
cat $PWD/config.rz.yml | sed "s/ARCH/$ARCH/g;s/SIZE/$SIZE/g" > $PWD/config.rz.$ARCH.$SIZE.yml
echo "CUDA_VISIBLE_DEVICES=0 nohup python3 /home/apd10/ExtremeClassificationRz/train.py --tmpdir /home/apd10/ExtremeClassificationRz/runs/exp1/  --config $PWD/config.rz.$ARCH.$SIZE.yml &" >> run.sh
SIZE=2000000,25000,25000,6500000
cat $PWD/config.rz.yml | sed "s/ARCH/$ARCH/g;s/SIZE/$SIZE/g" > $PWD/config.rz.$ARCH.$SIZE.yml
echo "CUDA_VISIBLE_DEVICES=0 nohup python3 /home/apd10/ExtremeClassificationRz/train.py --tmpdir /home/apd10/ExtremeClassificationRz/runs/exp1/  --config $PWD/config.rz.$ARCH.$SIZE.yml &" >> run.sh
SIZE=200000,25000,25000,650000
cat $PWD/config.rz.yml | sed "s/ARCH/$ARCH/g;s/SIZE/$SIZE/g" > $PWD/config.rz.$ARCH.$SIZE.yml
echo "CUDA_VISIBLE_DEVICES=0 nohup python3 /home/apd10/ExtremeClassificationRz/train.py --tmpdir /home/apd10/ExtremeClassificationRz/runs/exp1/  --config $PWD/config.rz.$ARCH.$SIZE.yml &" >> run.sh
SIZE=20000,10000,10000,65000
cat $PWD/config.rz.yml | sed "s/ARCH/$ARCH/g;s/SIZE/$SIZE/g" > $PWD/config.rz.$ARCH.$SIZE.yml
echo "CUDA_VISIBLE_DEVICES=0 nohup python3 /home/apd10/ExtremeClassificationRz/train.py --tmpdir /home/apd10/ExtremeClassificationRz/runs/exp1/  --config $PWD/config.rz.$ARCH.$SIZE.yml &" >> run.sh
