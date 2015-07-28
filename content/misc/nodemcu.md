Title: Reflashing NodeMCU devkit v2.0 on OS X
Date: 2015-07-27 20:57
Tags: NodeMCU, lua, EPS8266, hardware
Summary: How to reflash nodemcu-firmware to the NodeMCU devkit v2.0 on OS X

![NodeMCU v2.0](/images/206671f3b9855ed25d2ad8db98dd8e49.image.530x397.jpg)

This took me a long time to find between the different NodeMCU devkit versions, nodemcu's Delphi(??) windows-only flashing tool, some kind of competing python thing that comes in a .rar file but might not be windows-only anymore and doesn't work anyway, people writing Arduino firmware to the EPS8266, and the fact that the main communication medium in 2015 for this sort of thing is somehow still web forums.

So here it is. Download [esptool](https://github.com/themadinventor/esptool) and find nodemcu_float_0.9.6-dev_20150704.bin

    ./esptool.py --port /dev/tty.SLAB_USBtoUART write_flash \
        -fm dio -fs 32m 0x00000 nodemcu_float_0.9.6-dev_20150704.bin
