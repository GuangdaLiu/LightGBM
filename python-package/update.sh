sudo python3 setup.py install
sudo rm /usr/local/lib/python3.8/dist-packages/lightgbm/lib_lightgbm.so
sudo cp /home/guangdaliu/LightGBM/python-package/compile/lib_lightgbm.so /usr/local/lib/python3.8/dist-packages/lightgbm