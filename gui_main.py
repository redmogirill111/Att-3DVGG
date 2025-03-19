import os
import threading
import time
import tkinter.messagebox as msgbox
from tkinter import *
import cv2
import numpy as np
from tkinter.scrolledtext import ScrolledText
from PIL import Image as pilImage
from PIL import ImageTk as pilImageTk
import infer

# import pyaudio
import wave

def play_sound(sfile):
    chunk = 1024
    wf = wave.open(sfile, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    data = wf.readframes(chunk)

    while len(data) > 0:
        stream.write(data)
        data = wf.readframes(chunk)

    stream.stop_stream()
    stream.close()

    p.terminate()


class GUI():
    def __init__(self):

        self.infer_network=None
        self.root=Tk()
        self.root.title("多模态烟火识别系统")
        self.LEFT_panel=Frame(self.root)
        self.MID_panel=Frame(self.root)
        self.RIGHT_panel=Frame(self.root)
        self.LEFT_panel.pack(side=LEFT)
        self.MID_panel.pack(side=LEFT)
        self.RIGHT_panel.pack(side=LEFT)
        self.build_LEFT()
        self.build_middle()
        self.build_right()
        self.stop_work_flag=False
        self.double_mod_rgb_addr=''
        self.double_mod_nir_addr=''
        self.video_path=''
        self.single_mod_rgb_addr=''
        self.algorithm1_py_path=''
        self.algorithm2_py_path = ''
        self.algorithm1_model_path = ''
        self.algorithm2_model_path = ''

        self.alarm_script_file=''
        self.sound_file_path=''
        self.script_file_path=''
        self.double_mod_kjgbzd=None
        self.double_mod_jhwbzd = None
        self.root.mainloop()


    def build_LEFT(self):
        self.logo=Label(self.LEFT_panel,image=PhotoImage(file="G:\program\Fire-Detection-Base-3DCNN\gui\logo2_120.gif"))
        # Todo logo图片未显示
        # self.logo=Label(self.LEFT_panel)
        self.logo.pack(side=TOP)


        self.double_model_btn=Button(self.LEFT_panel,text='双模态',width=14,height=2,command=self.double_mod_file_chose)
        self.double_model_btn.pack(side=TOP)
        self.single_model_btn = Button(self.LEFT_panel, text='单模态',width=14,height=2,command=self.single_mod_file_chose)
        self.single_model_btn.pack(side=TOP)
        self.video_file_btn=Button(self.LEFT_panel, text='视频文件',width=14,height=2,command=self.video_file_chose)
        self.video_file_btn.pack(side=TOP)
        Label(self.LEFT_panel,width=10,height=2).pack(side=TOP)
        self.algorithm_1_btn=Button(self.LEFT_panel, text='算法1',width=14,height=2,command=self.chose_algor1_file)
        self.algorithm_1_btn.pack(side=TOP)
        self.algorithm_2_btn = Button(self.LEFT_panel, text='算法2',width=14,height=2,command=self.chose_algor2_file)
        self.algorithm_2_btn.pack(side=TOP)
        Label(self.LEFT_panel,width=10,height=2).pack(side=TOP)
        self.sound_file_chosen_btn=Button(self.LEFT_panel, text='告警声音',width=14,height=2,command=self.chose_alarm_sound_file)
        self.sound_file_chosen_btn.pack(side=TOP)
        self.alarm_script_chosen_btn = Button(self.LEFT_panel, text='告警脚本',width=14,height=2,command=self.chose_alarm_script_file)
        self.alarm_script_chosen_btn.pack(side=TOP)
        Label(self.LEFT_panel, width=10, height=10).pack(side=TOP)

    def build_middle(self):
        self.top=Label(self.MID_panel)
        self.top.pack(side=TOP)

        self.L_infer_Label=Label(self.top,text='当前推理标签:',height=2,width=20)
        self.L_infer_Label.pack(side=LEFT)
        self.infer_Label=Label(self.top,text='',height=2,width=8)
        self.infer_Label.pack(side=LEFT)
        Label(self.top,width=30).pack(side=LEFT)
        self.L_alarm_stats=Label(self.top,text='告警状态:',height=2,width=16)
        self.L_alarm_stats.pack(side=LEFT)
        self.alarm_stats=Label(self.top,text='',height=2,width=8)
        self.alarm_stats.pack(side=LEFT)

        self.image_panel=Label(self.MID_panel)
        self.image_panel.pack(side=TOP)

        self.show_black()
        Label(self.MID_panel, text='日志：').pack(side=TOP)
        self.detect_log=ScrolledText(self.MID_panel, width=88, height=10)
        self.detect_log.pack(side=TOP)
    def build_right(self):
        alarm_t=Label(self.RIGHT_panel)
        alarm_t.pack(side=TOP)
        self.L_alarm_delay_thr=Label(alarm_t,text='报警延迟阈值(秒)  ')
        self.L_alarm_delay_thr.pack(side=LEFT)
        self.alarm_delay_thr_v=Entry(alarm_t,width=5)
        self.alarm_delay_thr_v.pack(side=LEFT)

        alarm_c=Label(self.RIGHT_panel)
        alarm_c.pack(side=TOP)
        self.L_alarm_delay_c_thr=Label(alarm_c,text='报警延迟阈值(次数)')
        self.L_alarm_delay_c_thr.pack(side=LEFT)
        self.alarm_delay_c_thr_v=Entry(alarm_c,width=5)
        self.alarm_delay_c_thr_v.pack(side=LEFT)

        Label(self.RIGHT_panel,height=5).pack(side=TOP)

        choose_input=Label(self.RIGHT_panel)
        choose_input.pack(side=TOP)
        OptionList_choose_input=['单模态','双模态','视频文件']
        L_choose_input=Label(choose_input,text='选择输入：')
        L_choose_input.pack(side=LEFT)
        self.variable_choose_input = StringVar(self.RIGHT_panel)
        self.variable_choose_input.set("下拉选择")
        opt_choose_input = OptionMenu(choose_input, self.variable_choose_input, *OptionList_choose_input)
        opt_choose_input.config(width=8)
        opt_choose_input.pack(side=LEFT)

        choose_arch=Label(self.RIGHT_panel)
        choose_arch.pack(side=TOP)
        OptionList_choose_arch=['算法1','算法2']
        L_choose_arch=Label(choose_arch,text='选择算法：')
        L_choose_arch.pack(side=LEFT)
        self.variable_choose_arch = StringVar(self.RIGHT_panel)
        self.variable_choose_arch.set("下拉选择")
        opt_choose_arch = OptionMenu(choose_arch, self.variable_choose_arch, *OptionList_choose_arch)
        opt_choose_arch.config(width=8)
        opt_choose_arch.pack(side=LEFT)

        choose_alarm_delay=Label(self.RIGHT_panel)
        choose_alarm_delay.pack(side=TOP)
        OptionList_choose_alarm_delay=['不延时','时间延时','阈值延时']
        L_choose_alarm_delay=Label(choose_alarm_delay,text='报警延时：')
        L_choose_alarm_delay.pack(side=LEFT)
        self.variable_choose_alarm_delay = StringVar(self.RIGHT_panel)
        self.variable_choose_alarm_delay.set("下拉选择")
        opt_choose_alarm_delay = OptionMenu(choose_alarm_delay, self.variable_choose_alarm_delay, *OptionList_choose_alarm_delay)
        opt_choose_alarm_delay.config(width=8)
        opt_choose_alarm_delay.pack(side=LEFT)

        choose_alarm_mothed=Label(self.RIGHT_panel)
        choose_alarm_mothed.pack(side=TOP)
        OptionList_choose_alarm_mothed=['不告警','告警声音','告警脚本','声音+脚本']
        L_choose_alarm_mothed=Label(choose_alarm_mothed,text='报警方式：')
        L_choose_alarm_mothed.pack(side=LEFT)
        self.variable_choose_alarm_mothed = StringVar(self.RIGHT_panel)
        self.variable_choose_alarm_mothed.set("下拉选择")
        opt_choose_alarm_mothed = OptionMenu(choose_alarm_mothed, self.variable_choose_alarm_mothed, *OptionList_choose_alarm_mothed)
        opt_choose_alarm_mothed.config(width=8)
        opt_choose_alarm_mothed.pack(side=LEFT)

        Label(self.RIGHT_panel,height=6).pack(side=TOP)

        L_btn=Label(self.RIGHT_panel)
        L_btn.pack(side=TOP)
        self.start_work_btn=Button(L_btn,text='开始监测',width=12,command=self.start_work)
        self.start_work_btn.pack(side=LEFT)
        self.stot_work_btn = Button(L_btn, text='停止监测',width=12,command=self.stop_work)
        self.stot_work_btn.pack(side=LEFT)
        L_r=Label(self.RIGHT_panel)
        L_r.pack(side=TOP)
        Label(L_r,width=5)
        self.reload_btn = Button(L_r, text='重载系统',width=12,command=self.reload)
        self.reload_btn.pack(side=LEFT)

    def reload(self):
        if self.variable_choose_arch.get()=='算法1':
            model_py_file=self.algorithm1_py_path
            model_file=self.algorithm1_model_path
        elif self.variable_choose_arch.get()=='算法2':
            model_py_file = self.algorithm2_py_path
            model_file = self.algorithm2_model_path
        else:
            print('error')
            return
        if self.variable_choose_input.get()=='单模态':
            video_input=self.single_mod_rgb_addr

        elif self.variable_choose_input.get()=='双模态':
            video_input = self.double_mod_rgb_addr,self.double_mod_nir_addr
            bzd=self.double_mod_kjgbzd,self.double_mod_jhwbzd
        elif self.variable_choose_input.get()=='视频文件':
            video_input = self.video_path
        else:
            print('error2')
            return

        if self.variable_choose_input.get()=='双模态':
            self.infer_network=infer.network(model_py_file,model_file,video_input,bzd,True)
        else:
            self.infer_network = infer.network(model_py_file, model_file, video_input)

        pass


    def stop_work(self):
        self.show_black()
        self.infer_network.stop_flag=True
        pass
    def start_work(self):
        print(self.variable_choose_input.get())
        print(self.variable_choose_arch.get())
        print(self.variable_choose_alarm_mothed.get())
        print(self.variable_choose_alarm_delay.get())
        if self.variable_choose_arch.get()=='算法1':
            model_py_file=self.algorithm1_py_path
            model_file=self.algorithm1_model_path
        elif self.variable_choose_arch.get()=='算法2':
            model_py_file = self.algorithm2_py_path
            model_file = self.algorithm2_model_path
        else:
            print('未正确选择算法')
            return
        if self.variable_choose_input.get()=='单模态':
            video_input=self.single_mod_rgb_addr

        elif self.variable_choose_input.get()=='双模态':
            video_input = self.double_mod_rgb_addr,self.double_mod_nir_addr
            bzd=self.double_mod_kjgbzd,self.double_mod_jhwbzd
        elif self.variable_choose_input.get()=='视频文件':
            video_input = self.video_path
        else:
            print('未正确选择模态')
            return

        if self.variable_choose_input.get()=='双模态':
            if (model_py_file == "") or (model_file == "") or (video_input == "") or (bzd == ""):
                print("未输入模型文件或是视频参数")
            print(model_py_file, model_file, video_input,bzd)
            self.infer_network=infer.network(model_py_file,model_file,video_input,bzd,True)

        else:
            if (model_py_file == "") or (model_file == "") or (video_input == ""):
                print("未输入模型文件或是视频文件")
            print(model_py_file, model_file, video_input)
            self.infer_network = infer.network(model_py_file, model_file, video_input)


        threading.Thread(target=self.infer_network.get_frame, args=()).start()
        threading.Thread(target=self.infer_network.detect,args=()).start()
        threading.Thread(target=self.alarm_thr,args=()).start()
        self.show_image_and_info()
        # print(model_py_file,model_file,video_input,self.variable_choose_input.get()=='双模态')
        pass

    def chose_alarm_script_file(self):
        top = Toplevel()
        p1 = Label(top)
        p1.pack(side=TOP)

        def getpath():
            from tkinter import filedialog
            E_script.insert(0,filedialog.askopenfilename())

        def comfirm_and_close(self):
            self.script_file_path = E_script.get()
            top.destroy()
            # return rgb_addr,nir_addr

        p2 = Label(top)
        p2.pack(side=TOP)
        Label(p1, text='脚本文件路径：').pack(side=LEFT)
        E_script = Entry(p1,width=50)
        E_script.pack(side=LEFT)
        Button(p1, text='浏览文件', command=getpath).pack(side=LEFT)
        Button(p2, text='确认', command=lambda: comfirm_and_close(self)).pack(side=LEFT)


    def alarm_thr(self):
        while True:
            if self.infer_network.stop_flag:
                break
            time.sleep(0.5)
            if self.infer_network.mess_q.is_empty():
                continue
            lab=self.infer_network.mess_q.dequeue()
            if lab=='fire':
                if self.variable_choose_alarm_delay=='不延时':
                    if self.variable_choose_alarm_mothed=='不告警':
                        self.alarm_stats['text'] = ''
                        pass
                    elif self.variable_choose_alarm_mothed=='告警声音':
                        play_sound(self.sound_file_path)
                        self.alarm_stats['text'] = '告警'
                        pass
                    elif self.variable_choose_alarm_mothed == '告警脚本':
                        os.system(self.script_file_path)
                        self.alarm_stats['text'] = '告警'
                        pass
                    elif self.variable_choose_alarm_mothed == '声音+脚本':
                        play_sound(self.sound_file_path)
                        os.system(self.script_file_path)
                        self.alarm_stats['text']='告警'
                        pass

                elif self.variable_choose_alarm_delay=='时间延时':
                    st=time.time()
                    flag=False
                    while True:
                        if time.time()-st > int(self.alarm_delay_thr_v.get()):
                            flag=True
                            break
                        else:
                            if self.infer_network.mess_q.is_empty():
                                continue
                            if self.infer_network.mess_q.dequeue()=='fire':
                                pass
                            else:
                                break
                    if flag==True:
                        if self.variable_choose_alarm_mothed == '不告警':
                            self.alarm_stats['text'] = ''
                            pass
                        elif self.variable_choose_alarm_mothed == '告警声音':
                            play_sound(self.sound_file_path)
                            self.alarm_stats['text'] = '告警'
                            pass
                        elif self.variable_choose_alarm_mothed == '告警脚本':
                            os.system(self.script_file_path)
                            self.alarm_stats['text'] = '告警'
                            pass
                        elif self.variable_choose_alarm_mothed == '声音+脚本':
                            play_sound(self.sound_file_path)
                            os.system(self.script_file_path)
                            self.alarm_stats['text'] = '告警'
                            pass


                elif self.variable_choose_alarm_delay=='阈值延时':
                    c=1
                    flagc=False
                    while True:
                        if self.infer_network.mess_q.is_empty():
                            continue
                        if self.infer_network.mess_q.dequeue() == 'fire':
                            c=c+1
                            pass
                        else:
                            break
                        if c>int(self.alarm_delay_c_thr_v.get()):
                            flagc=True
                            break
                    if flagc:
                        if self.variable_choose_alarm_mothed == '不告警':
                            self.alarm_stats['text'] = ''
                            pass
                        elif self.variable_choose_alarm_mothed == '告警声音':
                            play_sound(self.sound_file_path)
                            self.alarm_stats['text'] = '告警'
                            pass
                        elif self.variable_choose_alarm_mothed == '告警脚本':
                            os.system(self.script_file_path)
                            self.alarm_stats['text'] = '告警'
                            pass
                        elif self.variable_choose_alarm_mothed == '声音+脚本':
                            play_sound(self.sound_file_path)
                            os.system(self.script_file_path)
                            self.alarm_stats['text'] = '告警'
            else:
                self.alarm_stats['text'] = ''

    def show_image_and_info(self):
        if self.infer_network.ret:
            # print('读取视频成功')
            img=self.infer_network.image
            cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)  # 转换颜色从BGR到RGBA
            cv2image = cv2.resize(cv2image,(640,480))
            current_image = pilImage.fromarray(cv2image)  # 将图像转换成Image对象
            imgtk = pilImageTk.PhotoImage(image=current_image)  # 显示图像
            self.image_panel.imgtk = imgtk
            self.image_panel.config(image=imgtk)

            if self.infer_network.infer_label:
                self.infer_Label['text']=self.infer_network.infer_label
        if self.infer_network.end or self.infer_network.stop_flag:
            self.show_black()
            msgbox.showinfo('','视频已结束')
        else:
            self.root.after(1,self.show_image_and_info)




    def chose_alarm_sound_file(self):
        top = Toplevel()
        p1 = Label(top)
        p1.pack(side=TOP)

        def getpath():
            from tkinter import filedialog
            E_sound.insert(0,filedialog.askopenfilename())

        def comfirm_and_close(self):
            self.sound_file_path = E_sound.get()
            top.destroy()
            # return rgb_addr,nir_addr

        p2 = Label(top)
        p2.pack(side=TOP)
        Label(p1, text='声音文件路径：').pack(side=LEFT)
        E_sound_def = StringVar(
            value=r'F:/THHI/program/Fire-Detection-Base-3DCNN/gui/chuxianyanhuo_15_4_4.mp3')
        E_sound = Entry(p1,width=50, textvariable=E_sound_def)
        E_sound.pack(side=LEFT)
        Button(p1, text='浏览文件', command=getpath).pack(side=LEFT)
        Button(p2, text='确认', command=lambda: comfirm_and_close(self)).pack(side=LEFT)



    def show_black(self):
        img = np.zeros((480,640,3),np.uint8)
        cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)  # 转换颜色从BGR到RGBA
        current_image = pilImage.fromarray(cv2image)  # 将图像转换成Image对象
        imgtk = pilImageTk.PhotoImage(image=current_image)#显示图像
        self.image_panel.imgtk = imgtk
        self.image_panel.config(image=imgtk)


    def double_mod_file_chose(self):
        top=Toplevel()
        p1=Label(top)
        p1.pack(side=TOP)

        def comfirm_and_close(self):
            self.double_mod_rgb_addr=E_rgb.get()
            self.double_mod_nir_addr=E_nir.get()
            from ast import literal_eval
            self.double_mod_kjgbzd=literal_eval(E_kjgbzd.get())
            self.double_mod_jhwbzd=literal_eval(E_jhwbzd.get())
            top.destroy()
            # return rgb_addr,nir_addr

        p2=Label(top)
        p2.pack(side=TOP)

        p3=Label(top)
        p3.pack(side=TOP)

        p4=Label(top)
        p4.pack(side=TOP)

        p5=Label(top)
        p5.pack(side=TOP)

        Label(p1,text='RGB的RTSP视频流地址：').pack(side=LEFT)
        E_rgb_def = StringVar(value='http://rtsp.zmgdcm.cn:9850/playServer/acquirePlayService?type=live&resourceId=1000000000000001&protocol=hls0&drmType=none&deviceGroup=TV(STB)&op=sovp&playType=catchup&redirect.m3u8')
        E_rgb=Entry(p1,width=50,textvariable=E_rgb_def)
        E_rgb.pack(side=LEFT)

        Label(p2, text='红外的RTSP视频流地址：').pack(side=LEFT)
        E_nir_def = StringVar(
            value='http://rtsp.zmgdcm.cn:9850/playServer/acquirePlayService?type=live&resourceId=1000000000000001&protocol=hls0&drmType=none&deviceGroup=TV(STB)&op=sovp&playType=catchup&redirect.m3u8')

        E_nir = Entry(p2,width=50,textvariable=E_nir_def)
        E_nir.pack(side=LEFT)

        Label(p3, text='可见光标志点：').pack(side=LEFT)
        E_kjgbzd_def = StringVar(
            value='[[399, 159],[109, 940],[1448, 1036], [1106, 870], [935, 583],    [1239, 632],    [1431, 306]]')
        E_kjgbzd = Entry(p3, width=60,textvariable=E_kjgbzd_def)
        E_kjgbzd.pack(side=LEFT)

        Label(p4, text='近红外标志点：').pack(side=LEFT)
        E_jhwbzd_def = StringVar(
            value='[[471, 145],[155, 929],[1512, 1034],[1172, 865],[1007, 575],   [1312, 625],    [1510, 298]]')
        E_jhwbzd = Entry(p4, width=60,textvariable=E_jhwbzd_def)
        E_jhwbzd.pack(side=LEFT)

        Button(p5,text='确认',command=lambda:comfirm_and_close(self)).pack(side=LEFT)

    def single_mod_file_chose(self):

        top=Toplevel()
        p1=Label(top)
        p1.pack(side=TOP)

        def comfirm_and_close(self):
            self.single_mod_rgb_addr=E_rgb.get()
            top.destroy()
            # return rgb_addr,nir_addr

        p2=Label(top)
        p2.pack(side=TOP)
        Label(p1,text='RGB的RTSP视频流地址：').pack(side=LEFT)
        E_rgb_def = StringVar(
            value='http://rtsp.zmgdcm.cn:9850/playServer/acquirePlayService?type=live&resourceId=1000000000000001&protocol=hls0&drmType=none&deviceGroup=TV(STB)&op=sovp&playType=catchup&redirect.m3u8')
        E_rgb=Entry(p1,width=50,textvariable=E_rgb_def)
        E_rgb.pack(side=LEFT)

        Button(p2,text='确认',command=lambda:comfirm_and_close(self)).pack(side=LEFT)

    def video_file_chose(self):
        top = Toplevel()
        p1 = Label(top)
        p1.pack(side=TOP)

        def getpath():
            from tkinter import filedialog
            E_rgb.insert(0,filedialog.askopenfilename())

        def comfirm_and_close(self):
            self.video_path = E_rgb.get()
            top.destroy()
            # return rgb_addr,nir_addr

        p2 = Label(top)
        p2.pack(side=TOP)
        Label(p1, text='视频文件路径：').pack(side=LEFT)
        E_rgb = Entry(p1,width=50)
        E_rgb.pack(side=LEFT)
        Button(p1, text='浏览文件', command=getpath).pack(side=LEFT)
        Button(p2, text='确认', command=lambda: comfirm_and_close(self)).pack(side=LEFT)

    def chose_algor1_file(self):
        top=Toplevel()
        p1=Label(top)
        p1.pack(side=TOP)

        def get_py_path():
            from tkinter import filedialog
            E_py.insert(0,filedialog.askopenfilename())

        def get_model_path():
            from tkinter import filedialog
            E_model.insert(0,filedialog.askopenfilename())

        def comfirm_and_close(self):
            self.algorithm1_py_path=E_py.get()
            self.algorithm1_model_path=E_model.get()
            top.destroy()
            # return rgb_addr,nir_addr

        p2=Label(top)
        p2.pack(side=TOP)
        Label(p1,text='算法1 py文件').pack(side=LEFT)
        E_py_def = StringVar(
            value=r'F:/THHI/program/Fire-Detection-Base-3DCNN/gui/4tongdao-T2CC3D_model.py')
        E_py=Entry(p1,width=50,textvariable=E_py_def)
        E_py.pack(side=LEFT)
        Button(p1,text='浏览',command=get_py_path).pack(side=LEFT)

        Label(p2, text='算法1 模型文件').pack(side=LEFT)
        E_model_def = StringVar(
            value=r'F:/THHI/program/Fire-Detection-Base-3DCNN/gui/4tongdao-vgg_3D-ucf101_epoch-158_acc-0.9990.pth.tar')
        E_model = Entry(p2,width=50,textvariable=E_model_def)
        E_model.pack(side=LEFT)
        Button(p2,text='浏览',command=get_model_path).pack(side=LEFT)
        Button(top,text='确认',command=lambda:comfirm_and_close(self)).pack(side=TOP)

    def chose_algor2_file(self):
        top=Toplevel()
        p1=Label(top)
        p1.pack(side=TOP)

        def get_py_path():
            from tkinter import filedialog
            E_py.insert(0,filedialog.askopenfilename())

        def get_model_path():
            from tkinter import filedialog
            E_model.insert(0,filedialog.askopenfilename())

        def comfirm_and_close(self):
            self.algorithm2_py_path=E_py.get()
            self.algorithm2_model_path=E_model.get()
            top.destroy()
            # return rgb_addr,nir_addr

        p2=Label(top)
        p2.pack(side=TOP)
        Label(p1,text='算法2 py文件').pack(side=LEFT)
        E_py_def = StringVar(value=r'F:/THHI/program/Fire-Detection-Base-3DCNN/gui/3tongdao_T2CC3D_model.py')
        E_py=Entry(p1,width=50,textvariable=E_py_def)
        E_py.pack(side=LEFT)
        Button(p1,text='浏览',command=get_py_path).pack(side=LEFT)


        Label(p2, text='算法2 模型文件').pack(side=LEFT)
        E_model_def = StringVar(value=r'F:/THHI/program/Fire-Detection-Base-3DCNN/gui/3tongdao_vgg_3D-ucf101_epoch-99_acc-0.9971.pth.tar')
        E_model = Entry(p2,width=50,textvariable=E_model_def)
        E_model.pack(side=LEFT)
        Button(p2,text='浏览',command=get_model_path).pack(side=LEFT)
        Button(top,text='确认',command=lambda:comfirm_and_close(self)).pack(side=TOP)


if __name__ == '__main__':
    GUI()