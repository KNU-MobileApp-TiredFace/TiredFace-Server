import serial
import os
from subprocess import call
from PIL import Image
import time

aruino = serial.Serial("/dev/cu.usbmodem1421",115200)
count = 0
first = 0

# while True:
#     op = input()
#     aruino.write(op.encode())

aruino.flush()

while True: 
    if aruino.readable():
        res = aruino.readline()
        res_string = res.decode('utf_8','ignore')[:len(res)-1]
        if res_string == "99922\r":
            main_ary = aruino.readline()
            first = 1
            file_name = str(count) + '.jpg'
            f = open(file_name,'wb')
            count += 1
            while True:
                if first == 1:
                    first = 0
                else:
                    if aruino.readable():
                        les = aruino.readline()
                        res_string = les.decode('utf_8','ignore')[:len(les)-1]
                        if res_string == "===CLOSE===\r":
                            f.write(main_ary)
                            f.close()

                            if count == 7:
                                cmd = "python main.py"
                                print('process call:', cmd)
                                cmd_args = cmd.split()
                                call(cmd_args)
                                exit()

                            for n in range(0,count):
                                if count > 2:
                                    if count-1 == n:
                                        temp = count-3
                                        cmd = "mv "+str(temp)+".jpg ./predict/"+str(temp)+".jpg"
                                        print('process call:', cmd)
                                        cmd_args = cmd.split()
                                        call(cmd_args)

                            # if count == 3:

                            # elif count == 4:
                            #     cmd = "mv 1.jpg ./predict/1.jpg"
                            #     print('process call:', cmd)
                            #     cmd_args = cmd.split()
                            #     call(cmd_args)
                            # elif count == 5:
                            #     cmd = "mv 2.jpg ./predict/2.jpg"
                            #     print('process call:', cmd)
                            #     cmd_args = cmd.split()
                            #     call(cmd_args)
                            # elif count == 6:    
                            #     cmd = "python main.py"
                            #     print('process call:', cmd)
                            #     cmd_args = cmd.split()
                            #     call(cmd_args)
                            #     exit()
                            break
                        main_ary += les
                        #print(les.decode('utf_8','ignore')[:len(les)-1])
                        #print(len(main_ary))


# while True:
#     if aruino.readable():
#         res = aruino.readline()
#         print(res.decode()[:len(res)-1])
#         count += 1
#     if count == 10:

#         break