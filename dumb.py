# import tkinter as tk


# class App:
#     def __init__(self, root=None):
#         self.root = root
#         self.frame = tk.Frame(self.root)
#         self.frame.pack()
#         tk.Label(self.frame, text='Main page').pack()
#         tk.Button(self.frame, text='Go to Page 1',
#                   command=self.make_page_1).pack()
#         self.page_1 = Page_1(master=self.root, app=self)

#     def main_page(self):
#         self.frame.pack()

#     def make_page_1(self):
#         self.frame.pack_forget()
#         self.page_1.start_page()


# class Page_1:
#     def __init__(self, master=None, app=None):
#         self.master = master
#         self.app = app
#         self.frame = tk.Frame(self.master)
#         tk.Label(self.frame, text='Page 1').pack()
#         tk.Button(self.frame, text='Go back', command=self.go_back).pack()

#     def start_page(self):
#         self.frame.pack()

#     def go_back(self):
#         self.frame.pack_forget()
#         self.app.main_page()


# if __name__ == '__main__':
#     root = tk.Tk()
#     app = App(root)
#     root.mainloop()
# Multi-frame tkinter application v2.3
import tkinter as tk

class SampleApp(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self._frame = None
        self.switch_frame(StartPage)

    def switch_frame(self, frame_class):
        """Destroys current frame and replaces it with a new one."""
        new_frame = frame_class(self)
        if self._frame is not None:
            self._frame.destroy()
        self._frame = new_frame
        self._frame.pack()

class StartPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        tk.Label(self, text="This is the start page").pack(side="top", fill="x", pady=10)
        tk.Button(self, text="Open page one",
                  command=lambda: master.switch_frame(PageOne)).pack()
        tk.Button(self, text="Open page two",
                  command=lambda: master.switch_frame(PageTwo)).pack()

class PageOne(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        tk.Label(self, text="This is page one").pack(side="top", fill="x", pady=10)
        tk.Button(self, text="Return to start page",
                  command=lambda: master.switch_frame(StartPage)).pack()

class PageTwo(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        tk.Label(self, text="This is page two").pack(side="top", fill="x", pady=10)
        tk.Button(self, text="Return to start page",
                  command=lambda: master.switch_frame(StartPage)).pack()

if __name__ == "__main__":
    app = SampleApp()
    app.mainloop()