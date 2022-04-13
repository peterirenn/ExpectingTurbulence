"""
Runs LSTM-TD3 algorithm via the machine class.
Communicates with experimental aerodynamic testbed via serial. 

Peter I Renn
"""
import os
import pickle
import tensorflow as tf
import numpy as np
from agent_TD3_keras import TD3
from MachineP import MachineP
from utils import KalmanFilter, NoneFilter
from datetime import datetime
import argparse
import serial
import nidaqmx
import time
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="machine for wing")
    parser.add_argument("-fil", "--filter", choices=["None", "Kalman"], default="None")
    args = parser.parse_args()

    # Initialize new machine object...
    machine = MachineP(**args.__dict__)
    machine._init(0)

    ## Number of Pressure Sensors
    n_sensors = 9
    # Time to run each episode for (in second)
    T = 1000
    # Time spent in episode
    t = 0
    # Number of steps to train for
    n_steps = 100
    # Max length of history to consider for LSTM
    max_hist_len = 10
    # Number of Episodes to run _request_stochastic_action
    n_episodes = 200

    ################################################
    #               SERIAL SETUP                   #
    ################################################
    # Create serial object
    ser = serial.Serial()
    # Set baudrate
    ser.baudrate = 115200
    # Set port
    ser.port = 'COM9'
    # Set timeout
    ser.timeout = 2
    # Open port
    ser.open()

    if ser.is_open != True:
        raise Exception("Serial Port not opened. ")

    ser.reset_input_buffer()

    try:
        # Read in and ignore everyting that you see to ensure buffer is clear
        while ser.in_waiting > 0:
            ser.reset_input_buffer()
            print(ser.in_waiting)
    except KeyboardInterrupt:
        ser.close()
        pass

    # Flush input buffer
    ser.reset_input_buffer()
    # Byte to initialize sensors - returned if sensors initialize properly.
    init_byte = b":"
    # Initialize the sensors
    ser.write(init_byte)
    # Check if enough time has passed for all sensors to return init_byte
    try:
        while ser.in_waiting < n_sensors:
            machine._stamp("Waiting to confirm sensors init...")
            machine._stamp("Number of sensors initialized: " + str(ser.in_waiting))
            time.sleep(3)
    except KeyboardInterrupt:
        ser.close()
        pass
    machine._stamp("Number of sensors initialized: " + str(ser.in_waiting))

    # Make sure each sensor is initialized.
    for i in range(n_sensors):
        check_init = ser.read()
        if check_init != init_byte:
            print(check_init)
            machine._stamp("Warning: Wrong number of sensors initialized...")
    machine._stamp("Sensors Initialized!")
    time.sleep(3)
    # Flush input buffer
    ser.reset_input_buffer()

    # Byte to zero sensors - returned if sensors zero proprely.
    zero_byte = b"+"
    # Make sure still open
    if ser.is_open != True:
        raise Exception("Serial Port not opened. ")
    # # Zero the Sensors
    ser.write(zero_byte)
    # Check if enough time has passed for all sensors to return init_byte
    try:
        while ser.in_waiting < n_sensors:
            machine._stamp("Waiting to confirm all sensors zeroed...")
            machine._stamp("Number of sensors zeroed: " + str(ser.in_waiting))
            # Zero the sensors
            time.sleep(10)
    except KeyboardInterrupt:
        ser.close()
        pass
    # Make sure that each sensor is zeroed
    for i in range(n_sensors):
        check_zero = ser.read()
        if check_zero != zero_byte:
            print(check_zero)
            ser.close()
    machine._stamp("All sensors zeroed!")

    ser.write(b";410")
    # Set speed
    ser.write(b"<816")
    sensor_mean = np.zeros((1,n_sensors))
    # Should return one set of data (For some reason?) - use that as zero
    for i in range(n_sensors):
        machine.state_mean[0,i] = ser.readline()
    time.sleep(2)

    # Flush input buffer of confirmation messages
    ser.reset_input_buffer()

    # Set up DAQ. Find the zero point.
    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan("Dev1/ai0")
        task.ai_channels.ai_range_high = 10
        task.ai_channels.ai_range_low = -10
        task.start()
        loads_to_mean = np.zeros(10)
        for l in range(10):
            loads_to_mean[l] = np.mean(task.read(1000))
        machine.load_mean[0,0] = np.mean(loads_to_mean)

        print(machine.load_mean[0,0])
        input("Press Enter to start...")
        dont_train = False
        try:
            while True:
                #####
                # Train episode
                #####

                # There is no error, yet...
                is_err = False
                # Tell pressure sensor to come back online
                ser.dtr = True
                # Flush input Serial
                ser.reset_input_buffer()
                # Start episode with known pressure voltage offset for mean state
                machine._start_episode(machine.state_mean, machine.load_mean)
                # Start reading data
                ser.write(b";410")
                # Make numpy array to hold state info
                raw_data = np.zeros((1, machine.state_dim))
                # Make numpy array to hold load info
                raw_load = np.zeros((1, 1))
                # Make buffers
                obs_buf = np.zeros((1, max_hist_len, machine.state_dim))
                act_buf = np.zeros((1, max_hist_len, machine.action_dim))
                # Track length of buffers
                bf_len = 0
                # Start counting...
                start = time.time()
                t_elapsed = 0
                counter = 0

                while counter < T:
                    counter = counter + 1
                    timer = time.time()
                    machine.time_record.append(t_elapsed)
                    # First element is the voltage from the loadcell
                    raw_load[0,0] = task.read()

                    try:
                        for i in range(n_sensors):
                            raw_data[0, i] = ser.readline()
                    except:
                        machine._stamp("Sensor #" + str(i) + "Fault")
                        machine._stamp("Unexpected Error: " + str(sys.exc_info()[0]))
                        # Check if there is an error
                        is_err = True
                        ser.dtr = False
                    # Request an action - action will be a number from -1 to 1
                    a = machine._request_stochastic_action(raw_data, raw_load, obs_buf, act_buf)

                    if bf_len < max_hist_len:
                        obs_buf[0, bf_len, :] = machine.state_record[-1]
                        act_buf[0, bf_len, :] = machine.action_record[-1]
                        bf_len = bf_len + 1
                    else:
                        obs_buf[0, :max_hist_len - 1, :] = obs_buf[0, 1:, :]
                        act_buf[0, :max_hist_len - 1, :] = act_buf[0, 1:, :]
                        obs_buf[0, max_hist_len - 1, :] = machine.state_record[-1]
                        act_buf[0, max_hist_len - 1, :] = machine.action_record[-1]
                    # Convert actions to a position ranging from -410 to 410 (correspons to [-40^o, +40^o])
                    servo_pos = int(np.rint(a[0] * 410) + 410)

                    # If it's more than 10 just attach it and send it.
                    if servo_pos >= 100:
                        servo_cmd = ";" + str(servo_pos)
                    elif servo_pos >= 10:
                        servo_cmd = ";0" + str(servo_pos)
                    else:
                        servo_cmd = ";00" + str(servo_pos)
                    ser.write(bytearray(servo_cmd,'utf-8'));
                    # Time stuff
                    t_elapsed = time.time() - timer
                    if (0.012 - t_elapsed > 0):
                        time.sleep(0.012 - t_elapsed)

                # Reset the position
                ser.write(b";410")
                # Tell pressure sensor to stop
                ser.dtr = False
                if machine.agent.episode_count == 201:
                    dont_train = True
                    T = 4000
                # If there is an error, skip training
                if is_err:
                    # Say what's happening..
                    machine._stamp('Restarting Episode ' + str(machine.agent.episode_count) + '!')
                    # So that the next "_start_episode" has the correct count
                    machine.agent.episode_count -= 1

                    # If there is an error, don't train or save, just restart
                    continue

                # Train
                if dont_train == False:
                    machine._train(n_steps)
                    # Save what you've done
                    machine._save()
                else:
                    machine._save()

                # There is no error, yet...
                is_err = False
                # Tell pressure sensor to come back online
                ser.dtr = True
                # Flush input Serial
                ser.reset_input_buffer()
                # Start episode with known pressure voltage offset for mean state
                machine._start_eval(machine.state_mean, machine.load_mean)
                # Start reading data
                ser.write(b";410")
                # Make numpy array to hold state info
                raw_data = np.zeros((1, machine.state_dim))
                # Make numpy array to hold load info
                raw_load = np.zeros((1, 1))
                # Make buffers
                obs_buf = np.zeros((1, max_hist_len, machine.state_dim))
                act_buf = np.zeros((1, max_hist_len, machine.action_dim))
                # Track length of buffers
                bf_len = 0
                # Start counting...
                start = time.time()
                t_elapsed = 0
                counter = 0
                # Make array to hold pressure before averaging
                #p_store = np.zeros((1,n_sensors))
                while counter < T:
                    counter = counter + 1
                    timer = time.time()
                    machine.time_record.append(t_elapsed)
                    # First element is the voltage from the loadcell
                    raw_load[0,0] = task.read()

                    # Rest is pressure values
                    try:
                        for i in range(n_sensors):
                            raw_data[0, i] = ser.readline()
                    except:
                        machine._stamp("Sensor #" + str(i) + "Fault")
                        machine._stamp("Unexpected Error: " + str(sys.exc_info()[0]))
                        # Check if there is an error
                        is_err = True
                        ser.dtr = False

                    # Request an action - I expect that the action will be a number from -1 to 1
                    a = machine._request_deterministic_action(raw_data, raw_load, obs_buf, act_buf)

                    if bf_len < max_hist_len:
                        obs_buf[0, bf_len, :] = machine.state_record[-1]
                        act_buf[0, bf_len, :] = machine.action_record[-1]
                        bf_len = bf_len + 1
                    else:
                        obs_buf[0, :max_hist_len - 1, :] = obs_buf[0, 1:, :]
                        act_buf[0, :max_hist_len - 1, :] = act_buf[0, 1:, :]
                        obs_buf[0, max_hist_len - 1, :] = machine.state_record[-1]
                        act_buf[0, max_hist_len - 1, :] = machine.action_record[-1]

                    # Convert actions to a position ranging from -410 to 410
                    servo_pos = int(np.rint(a[0] * 410) + 410)

                    # If it's more than 10 just attach it and send it.
                    if servo_pos >= 100:
                        servo_cmd = ";" + str(servo_pos)
                    elif servo_pos >= 10:
                        servo_cmd = ";0" + str(servo_pos)
                    else:
                        servo_cmd = ";00" + str(servo_pos)
                    ser.write(bytearray(servo_cmd,'utf-8'));
                    # Time stuff
                    t_elapsed = time.time() - timer
                    if (0.012 - t_elapsed > 0):
                        time.sleep(0.012 - t_elapsed)
                # Reset the position
                ser.write(b";410")
                # Tell pressure sensor to stop
                ser.dtr = False
                machine._save_eval()
                machine._stamp('Saved eval ' + str(machine.agent.episode_count) + '!')
                # If there is an error, skip training
                if is_err:
                    # Say what's happening..
                    machine._stamp('Restarting Episode ' + str(machine.agent.episode_count) + '!')
                    # So that the next "_start_episode" has the correct count
                    machine.agent.episode_count -= 1

                    # If there is an error, don't train or save, just restart
                    continue

        except KeyboardInterrupt:
            ser.close()
            pass
        task.stop()
    """ """
    ser.close()
