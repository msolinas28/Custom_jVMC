import numpy as np

# Complex floating point
tCpx = np.complex128
# Real floating point
tReal = np.float64

import jax

from functools import partial
import collections

myPmapDevices = jax.devices()  # [myDevice]
myDeviceCount = len(myPmapDevices)
pmap_for_my_devices = partial(jax.pmap, devices=myPmapDevices)

def pmap_devices_updated(pmapDevices):
    if collections.Counter(pmapDevices) == collections.Counter(myPmapDevices):
        return False
    return True


def get_iterable(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    else:
        return (x,)


def set_pmap_devices(devices):
    devices = list(get_iterable(devices))
    global myPmapDevices
    global myDeviceCount
    global pmap_for_my_devices
    myPmapDevices = devices
    myDeviceCount = len(myPmapDevices)
    pmap_for_my_devices = partial(jax.pmap, devices=myPmapDevices)
    myDevice = myPmapDevices[0]


def device_count():
    return len(myPmapDevices)


def devices():
    return myPmapDevices
