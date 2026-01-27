import subprocess

print("\nTensorFlow GPU(s) detected:")
import tensorflow as tf
g = tf.config.list_physical_devices('GPU')
[print("GPU", i, "->", tf.config.experimental.get_device_details(x).get('device_name','(no details)')) for i,x in enumerate(g)]

print("\nWindows detected GPU(s):")
print(subprocess.check_output("wmic path win32_VideoController get name", shell=True).decode())
