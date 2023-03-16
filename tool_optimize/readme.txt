

./ncnnoptimize crnn.param crnn.bin newcrnn.param newcrnn.bin 1

./ncnnoptimize dbnet.param dbnet.bin newdbnet.param newdbnet.bin 1

ncnnoptimize.exe 【原模型param文件】【原模型bin文件】【新模型param文件】【新模型bin文件】1（0=float32,1=float16）（0=float32,1=float16）
