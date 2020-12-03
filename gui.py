import tkinter as tk 
from tkinter import messagebox
from tkinter import ttk 
from PIL import ImageTk, Image
import pytesseract
import cv2
import numpy as np
#import face_recognition
import pandas as pd
import os
import csv
import time
import datetime
import sys
import shutil
#from Detector import test_model


def startpage(container):
    label = tk.Label(container, text ="Automatic Door Unlock System", font = "Helvetica", foreground="#263942")
    label.config(font=("Helvetica", 15))
    label.place(x = 100,y = 10)
    
    def admin_clear_frame(frame):
        for widget in frame.winfo_children():
            widget.destroy()
        
        admin(frame)
    

    # opens the image 
    img = Image.open('static/door.png') 
    
    img = img.resize((180, 180), Image.ANTIALIAS) 
    # PhotoImage class is used to add image to widgets, icons etc 
    img = ImageTk.PhotoImage(img) 
        # create a label 
    panel = tk.Label(container, image = img) 
        # set the image as img  
    panel.image = img 
    panel.place(x = 250 , y = 80)
    ttk.Style().configure("TButton", padding=6, relief="flat",
            background="#ccc",foreground='green')
    button1 = ttk.Button(container, text ="Admin",command = lambda : admin_clear_frame(container))
    button1.place(x = 95, y = 110)
    button2 = ttk.Button(container, text ="Doorbell",command = lambda : doorbell())
    button2.place(x = 95,y = 210)





    


def admin(container):
    
    label = tk.Label(container, text ="Admin Portal", font = "Helvetica", foreground="#263942")
    label.config(font=("Helvetica", 15))
    label.place(x = 180,y = 20)
    
    img = Image.open('static/login.png') 
    img = img.resize((190, 190), Image.ANTIALIAS) 
    img = ImageTk.PhotoImage(img) 
    
    panel = tk.Label(container, image = img) 
    panel.image = img 
    panel.place(x = 230 , y = 80)
    ttk.Style().configure("TButton", padding=6, relief="flat",
            background="#ccc",foreground='green') 
    

    def user_list_clear_frame(frame):
        for widget in frame.winfo_children():
            widget.destroy()
        
        user_list(frame)

    def new_user_clear_frame(frame):
        for widget in frame.winfo_children():
            widget.destroy()
        
        new_user(frame)

    def back_menu(frame):
        for widget in frame.winfo_children():
            widget.destroy()
        
        startpage(frame)

    button1 = ttk.Button(container, text ="Existing Users",command = lambda : user_list_clear_frame(container))
    button1.place(x = 82, y = 90)
    button2 = ttk.Button(container, text ="Add new User",command = lambda : new_user_clear_frame(container))
    button2.place(x = 82,y = 180)

    button3 = ttk.Button(container, text ="Back",command = lambda : back_menu(container))
    button3.place(x = 82,y = 270)


def new_user(container):
    new_user = tk.StringVar()
    flag = tk.IntVar()
    flag.set(0)
    num_images = tk.IntVar()

    label = tk.Label(container, text ="New User Registeration", font = "Helvetica", foreground="#263942")
    label.config(font=("Helvetica", 15))
    label.place(x = 130,y = 20)
        
    name_label = tk.Label(container, text ="Name :", font = "Helvetica", foreground="#263942")
    name_label.config(font=("Helvetica", 12))
    name_label.place(x = 95,y = 90)

    def clear(frame):
        
        for widget in frame.winfo_children():
            widget.destroy()
        admin(frame)

    def check(container,name,flag,button1,button2,button3,num_images):
        data = pd.read_csv('User.csv')
        print(data)
        if(name in list(data.Name)):
            messagebox.showerror("Error","User already Exists")
            return
        create_dataset(container,name,flag,button1,button2,button3,num_images)
        return

    def build_model(name,button1,button2,button3):
        entry_name.delete(0,'end')
        
        train_model(name,button1,button2,button3)

    entry_name = tk.Entry(container,textvariable = new_user)
    entry_name.place(x = 165, y = 90)
    ttk.Style().configure("TButton", padding=6, relief="flat",
            background="#ccc",foreground='green') 
    
    
    button3 = ttk.Button(container, text ="Back",command = lambda : clear(container),state = tk.NORMAL) 
    button2 = ttk.Button(container, text ="Train dataset",state = tk.DISABLED,command = lambda : build_model(new_user.get(),button1,button2,button3))
    button1 = ttk.Button(container, text ="Create dataset",command = lambda : check(container,new_user.get(),flag,button1,button2,button3,num_images)) 
    
    button1.place(x = 310, y = 180)
    button2.place(x = 180,y = 180)
    button3.place(x = 50,y = 180)



def create_dataset(container,name,flag,button1,button2,button3,num_images):
    path = "./dataset/" + name
    num_of_images = 0
    detector = cv2.CascadeClassifier("static/haarcascade_frontalface_default.xml")
    try:
        os.makedirs(path)
    except:
        print('Directory Already Created')
    vid = cv2.VideoCapture(0)
    while True:
        ret, img = vid.read()
        new_img = None
        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = detector.detectMultiScale(image=grayimg, scaleFactor=1.1, minNeighbors=5)
        key = 0
        for x, y, w, h in face:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 2)
            cv2.putText(img, "Face Detected", (x, y-5), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 255))
            cv2.putText(img, str(str(num_of_images)+" images captured"), (x, y+h+20), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 255))
            new_img = img[y:y+h, x:x+w]
            cv2.imshow("FaceDetection", img)
            key = cv2.waitKey(1) & 0xFF
        try :
            cv2.imwrite(str(path+"/"+str(num_of_images)+name+".jpg"), new_img)
            num_of_images += 1
        except :
            pass
        
        if num_of_images > 300:
            break
    cv2.destroyAllWindows()
    print(num_of_images)
    button2['state'] = "normal"
    button3['state'] = 'disabled'
    button1['state'] = 'disabled'
    flag.set(1)
    num_images.set(num_of_images)
    print(flag.get())
    app.protocol("WM_DELETE_WINDOW",disable_event)
    s = f"Images captuared : {num_images.get()}"
    label1 = tk.Label(container, text = s, font = "Helvetica", foreground="red")
    label1.config(font=("Helvetica", 12))
    label1.place(x = 150,y = 250)
    return


def train_model(name,button1,button2,button3):
    path = os.path.join(os.getcwd()+"/dataset/"+name+"/")

    faces = []
    ids = []
    labels = []
    pictures = {}

    for root,dirs,files in os.walk(path):
        pictures = files


    for pic in pictures :
        imgpath = path+pic
        img = Image.open(imgpath).convert('L')
        imageNp = np.array(img, 'uint8')
        id = int(pic.split(name)[0])
        #names[name].append(id)
        faces.append(imageNp)
        ids.append(id)

    ids = np.array(ids)
    #Train and save classifier
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("./classifiers/"+name+"_classifier.xml")
    button2['state'] = 'disabled'
    button3['state'] = "normal"
    button1['state'] = "normal"
    app.protocol("WM_DELETE_WINDOW",close)

    data = pd.read_csv('User.csv')
    data.loc[len(data.Name)] = [name]
    data.set_index('Name',inplace=True)
    
    data.to_csv('User.csv')
    
    messagebox.showinfo("Notififcation","Succesfully Trained the model")


def delete_selected(frame,Lb1):
    a = Lb1.get(Lb1.curselection()).split(' ')
    path = os.getcwd()
    print("Path : ",path)
    path1 = path + f"\\dataset\\{a[1]}"
    path2 = path + f'\\classifiers\\{a[1]}_classifier.xml' 
    print(path,path1,path2)
    shutil.rmtree(path1)
    os.remove(path2)
    
    data = pd.read_csv('User.csv')
    print(data)
    new_data = data[data.Name != a[1]]
    print(new_data)
    new_data.set_index('Name',inplace = True)
    print("New Dataset : ",new_data)
    new_data.to_csv('User.csv')
    
    
        
    for widget in frame.winfo_children():
        widget.destroy()
        
    user_list(frame)


def user_list(container):
    label = tk.Label(container, text ="List of Existing Users", font = "Helvetica", foreground="#263942")
    label.config(font=("Helvetica", 15))
    label.place(x = 140,y = 20)
    names = []
    Lb1 = tk.Listbox(container,selectbackground = "lightblue",yscrollcommand = True,bg = "#ccc")

    data = pd.read_csv('User.csv')
    z = list(data.Name)
    print(z)
    for i in range(len(z)):
        Lb1.insert(i+1, f"{i+1}. {z[i]}")        
    
    Lb1.place(x = 90,y = 80)
    ttk.Style().configure("TButton", padding=6, relief="flat",
            background="#ccc",foreground='green') 
    
    def back_clear_frame(frame):
        for widget in frame.winfo_children():
            widget.destroy()
        
        admin(frame)

    

    button1 = ttk.Button(container, text ="Delete", 
							command = lambda : delete_selected(container,Lb1))
    button1.place(x = 300, y = 120)

    button1 = ttk.Button(container, text ="Back", 
							command = lambda : back_clear_frame(container))
    button1.place(x = 300, y = 180)

def doorbell():
        data = pd.read_csv("User.csv")
        names = list(data.Name)
        face_cascade = cv2.CascadeClassifier('./static/haarcascade_frontalface_default.xml')
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        cap = cv2.VideoCapture(0)
        
        for i in names:
            print(i)
            name = i
            recognizer.read(f"./classifiers/{name}_classifier.xml")
            pred = 0
            for i in range(50):
                ret, frame = cap.read()
                #default_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray,1.3,5)

                for (x,y,w,h) in faces:


                    roi_gray = gray[y:y+h,x:x+w]

                    id,confidence = recognizer.predict(roi_gray)
                    confidence = 100 - int(confidence)
                    
                    if confidence > 50:
                        #if u want to print confidence level
                                #confidence = 100 - int(confidence)
                                pred += +1
                                text = name.upper()
                                font = cv2.FONT_HERSHEY_PLAIN
                                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                frame = cv2.putText(frame, text, (x, y-4), font, 1, (0, 255, 0), 1, cv2.LINE_AA)
                                print("Matched Face")
                                
                                if(pred == 5):
                                    time_now = datetime.datetime.now()
                                    path = os.getcwd() + f"\\results\\{name}{time_now}.jpg"
                                    #print(frame)
                                    #print(path)
                                    s = ".\\results\\"+ str(name) + str(time_now.date()) + "-" + str(time_now.hour) + "-" +str(time_now.minute) + "-" +str(time_now.second)
                                    cv2.imwrite(s+".jpg", frame)
                                    cv2.waitKey(2000)
                                    cap.release()
                                    cv2.destroyAllWindows()
                                    excel_data = pd.read_excel('entries.xlsx')
                                    excel_data.loc[len(excel_data)] = [name,datetime.datetime.now()]
                                    excel_data.to_excel('entries.xlsx',index = False)
                                    messagebox.showinfo("Notification","User Detected Open the door")
                                    return    
                    else:   
                                #pred += -1
                                text = "UnknownFace"
                                font = cv2.FONT_HERSHEY_PLAIN
                                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                                frame = cv2.putText(frame, text, (x, y-4), font, 1, (0, 0,255), 1, cv2.LINE_AA)

                cv2.imshow("image", frame)


                if cv2.waitKey(20) & 0xFF == ord('q'):
                    print(pred)
                    
        messagebox.showerror("Error","Unauthorized Person doors are closed")

        cap.release()
        cv2.destroyAllWindows()

'''
def train_model(name):
    path = os.path.join(os.getcwd()+"/data/"+name+"/")

    faces = []
    ids = []
    labels = []
    pictures = {}


    # Store images in a numpy format and ids of the user on the same index in imageNp and id lists

    for root,dirs,files in os.walk(path):
            pictures = files


    for pic in pictures :

            imgpath = path+pic
            img = Image.open(imgpath).convert('L')
            imageNp = np.array(img, 'uint8')
            id = int(pic.split(name)[0])
            #names[name].append(id)
            faces.append(imageNp)
            ids.append(id)

    ids = np.array(ids)

    #Train and save classifier
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("./data/classifiers/"+name+"_classifier.xml")



def train_model(container,name,button3):
    img_train = face_recognition.load_image_file('train.png')
    img_train = cv2.cvtColor(img_train,cv2.COLOR_BGR2RGB)
    encodeTrain = face_recognition.face_encodings(img_train)[0]
    print(encodeTrain)
    button3['state'] = 'normal'
    np.save(f'dataset/{name}.npy',encodeTrain)
    messagebox.showinfo("Notification","Trained Dataset Succesfully")
    for widget in container.winfo_children():
            widget.destroy()
        
    admin(container)


def create_train(name):
    path = "./data/" + name
    num_of_images = 0
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    try:
        os.makedirs(path)
    except:
        print('Directory Already Created')
    vid = cv2.VideoCapture(0)
    while True:
        ret, img = vid.read()
        new_img = None
        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = detector.detectMultiScale(image=grayimg, scaleFactor=1.1, minNeighbors=5)
        for x, y, w, h in face:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 2)
            cv2.putText(img, "Face Detected", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
            cv2.putText(img, str(str(num_of_images)+" images captured"), (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
            new_img = img[y:y+h, x:x+w]
            cv2.imshow("FaceDetection", img)
            key = cv2.waitKey(1) & 0xFF
        try :
            cv2.imwrite(str(path+"/"+str(num_of_images)+name+".jpg"), new_img)
            num_of_images += 1
        except :
            pass
        key = 0
        if num_of_images > 310:
            break
    cv2.destroyAllWindows()
    print(num_of_images)
    return num_of_images


def create_dataset(button2,new_user,button3):
    global app
    print("Hey")
    print(button2)
    root = tk.Toplevel(app)
    root.geometry('330x330')
    ttk.Style().configure("TButton", padding=6, relief="flat",
            background="#ccc",foreground='green') 
    width, height = 300, 300
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    label = tk.Label(root)
    label.place(x = 0,y = 0)
    
    

    
        
    def capture(root,frame,new_user,button2,button3):
        print("Frame  is : ",frame)
        try:
            _, frame = cap.read()
            cv2.imwrite('train.png', frame)
            time.sleep(1)
            imgTest = face_recognition.load_image_file('train.png')
            imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)
            faceLocTest = face_recognition.face_locations(imgTest)[0]
        
        except:
            messagebox.showerror("Error","Unable to recognize the face")
            root.destroy()
            return
        button2['state'] = 'normal'
        button3['state'] = 'disabled'
        np.save(f'dataset/{new_user.get()}.npy',np.array(2))
        data = pd.read_csv('User.csv')
        data.loc[len(data.index)] = [new_user.get()]
        data.set_index('Name',inplace = True)
        data.to_csv('User.csv')
        
        root.destroy()
    
    def show_frame():
            _, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            label.imgtk = imgtk
            label.configure(image=imgtk)
            label.after(10, show_frame)

    _, frame = cap.read()
    
    show_frame()
    button1 = ttk.Button(root, text ="Capture",command = lambda : capture(root,frame,new_user,button2,button3)) 
    button1.place(x = 120, y = 250)    
    root.mainloop()

def new_user(container):
    new_user = tk.StringVar()
    status = tk.IntVar()
    status.set(0)
    label = tk.Label(container, text ="New User Registeration", font = "Helvetica", foreground="#263942")
    label.config(font=("Helvetica", 15))
    label.place(x = 130,y = 20)
        
    name_label = tk.Label(container, text ="Name :", font = "Helvetica", foreground="#263942")
    name_label.config(font=("Helvetica", 12))
    name_label.place(x = 95,y = 90)

    def clear(frame):
        for widget in frame.winfo_children():
            widget.destroy()
        admin(frame)
        
    entry_name = tk.Entry(container,textvariable = new_user)
    entry_name.place(x = 165, y = 90)
    ttk.Style().configure("TButton", padding=6, relief="flat",
            background="#ccc",foreground='green') 
    
    button3 = ttk.Button(container, text ="Back",command = lambda : clear(container),state = tk.NORMAL) 

    button2 = ttk.Button(container, text ="Train dataset",state = tk.DISABLED,command = lambda : train_model(container,new_user.get(),button3))
    

    #button1 = ttk.Button(container, text ="Create dataset",command = lambda : create_dataset(button2,new_user,button3)) 
    button1 = ttk.Button(container, text ="Create dataset",command = lambda : create_train(new_user.get())) 
    
    button1.place(x = 82, y = 180)

    button2.place(x = 210,y = 180)

    
    
    button3.place(x = 146, y = 260)







def doorbell(container):
    global app
    root = tk.Toplevel(app)
    root.geometry('330x330')
    ttk.Style().configure("TButton", padding=6, relief="flat",
            background="#ccc",foreground='green') 
    width, height = 300, 300
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    label = tk.Label(root)
    label.place(x = 0,y = 0)
    
    
    def show_frame():
            _, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            label.imgtk = imgtk
            label.configure(image=imgtk)
            label.after(10, show_frame)

    def test():
        try:
            _, frame = cap.read()
            cv2.imwrite('test.png', frame)
            time.sleep(1)
            imgTest = face_recognition.load_image_file('test.png')
            imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)
            faceLocTest = face_recognition.face_locations(imgTest)[0]
            encodeTest = face_recognition.face_encodings(imgTest)[0]

        except:
            print("Error")
            return    

        path = os.getcwd() + "\dataset"
        faces = []
        for files in os.listdir(path):
            if os.path.isfile(os.path.join(path,files)):
                print(files)
                encode = np.load(f'dataset/{files}')
                faces.append(encode)

        print(faces)

        results = face_recognition.compare_faces(faces,encodeTest)
        print('results : ',results)
        person = "unknown"
        if(True in results):
            messagebox.showinfo("Notification",'User Detected Open door')
            person_index = results.index(True)
            data = pd.read_csv('User.csv')
            person = list(data.Name)[person_index]
            

        else:
            messagebox.showerror("Error",'User not recognized door lock')
        print(person)
        excel_data = pd.read_excel('entries.xlsx')
        excel_data.loc[len(excel_data)] = [person,datetime.datetime.now()]
        excel_data.to_excel('entries.xlsx',index = False)
        root.destroy()



    _, frame = cap.read()
    
    show_frame()
    
    root.after(2000,test)
    root.mainloop()

'''


app = tk.Tk()
app.geometry("450x350") 
app.resizable(False,False)
container = tk.Frame(app)
container.pack(side = "top", fill = "both", expand = True)
container.grid_rowconfigure(0, weight = 1)
container.grid_columnconfigure(0, weight = 1)
startpage(container)

def close():
    app.destroy()

def disable_event():
    pass
app.mainloop()