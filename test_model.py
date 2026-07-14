#!/usr/bin/env python3
"""
BLE WiFi Provisioning - Pi side (BLE Peripheral / Server)
Using 'bless' library - compatible with modern BlueZ (5.65+)
"""
import asyncio
import subprocess
from bless import (
    BlessServer,
    GATTCharacteristicProperties,
    GATTAttributePermissions,
)

WIFI_SERVICE_UUID    = '12345678-0000-1000-8000-00805f9b34fb'
SSID_CHAR_UUID       = '12345678-0001-1000-8000-00805f9b34fb'
PASSWORD_CHAR_UUID   = '12345678-0002-1000-8000-00805f9b34fb'
STATUS_CHAR_UUID     = '12345678-0003-1000-8000-00805f9b34fb'
DEVICEINFO_CHAR_UUID = '12345678-0004-1000-8000-00805f9b34fb'

wifi_creds = {'ssid': None, 'password': None}
server = None


def connect_to_wifi(ssid, password):
    print(f"[WiFi] Attempting to connect to '{ssid}'...")
    set_status(1)
    try:
        result = subprocess.run(
            ['sudo', 'nmcli', 'dev', 'wifi', 'connect', ssid, 'password', password],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            print("[WiFi] Connected successfully!")
            set_status(2)
        else:
            print(f"[WiFi] Failed: {result.stderr}")
            set_status(3)
    except Exception as e:
        print(f"[WiFi] Error: {e}")
        set_status(3)


def set_status(value):
    char = server.get_characteristic(STATUS_CHAR_UUID)
    char.value = bytearray([value])
    print(f"[BLE] Status updated to: {value}")


def read_request(characteristic, **kwargs):
    print(f"[BLE] Read request on {characteristic.uuid}")
    if characteristic.uuid == DEVICEINFO_CHAR_UUID:
        return "RaspberryPi3-CamNode-v1".encode('utf-8')
    return characteristic.value


def write_request(characteristic, value, **kwargs):
    text_value = value.decode('utf-8').strip()
    print(f"[BLE] Write on {characteristic.uuid}: {text_value}")

    if characteristic.uuid == SSID_CHAR_UUID:
        wifi_creds['ssid'] = text_value
        characteristic.value = value
        print(f"[BLE] Received SSID: {text_value}")

    elif characteristic.uuid == PASSWORD_CHAR_UUID:
        wifi_creds['password'] = text_value
        characteristic.value = value
        print(f"[BLE] Received Password: {'*' * len(text_value)}")
        if wifi_creds['ssid']:
            connect_to_wifi(wifi_creds['ssid'], wifi_creds['password'])


async def run():
    global server
    server = BlessServer(name="PiCam")
    server.read_request_func = read_request
    server.write_request_func = write_request

    await server.add_new_service(WIFI_SERVICE_UUID)

    write_perm = GATTAttributePermissions.writeable
    read_perm = GATTAttributePermissions.readable

    await server.add_new_characteristic(
        WIFI_SERVICE_UUID, SSID_CHAR_UUID,
        GATTCharacteristicProperties.write,
        None, write_perm
    )
    await server.add_new_characteristic(
        WIFI_SERVICE_UUID, PASSWORD_CHAR_UUID,
        GATTCharacteristicProperties.write,
        None, write_perm
    )
    await server.add_new_characteristic(
        WIFI_SERVICE_UUID, STATUS_CHAR_UUID,
        GATTCharacteristicProperties.read | GATTCharacteristicProperties.notify,
        bytearray([0x00]), read_perm
    )
    await server.add_new_characteristic(
        WIFI_SERVICE_UUID, DEVICEINFO_CHAR_UUID,
        GATTCharacteristicProperties.read,
        "RaspberryPi3-CamNode-v1".encode('utf-8'), read_perm
    )

    try:
        await server.start()
        print("[BLE] Advertising as 'PiCam'... waiting for phone to connect")
    except Exception as e:
        print(f"[BLE] Built-in advertising failed ({e}), continuing anyway - using manual advertising instead")

    while True:
        await asyncio.sleep(1)


if __name__ == '__main__':
    asyncio.run(run())