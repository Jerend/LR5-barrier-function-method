import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import csv
from datetime import datetime
import os
import subprocess

class BarrierMethodApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Метод барьерных функций")
        self.root.geometry("1600x1000")
        
        # Параметры по умолчанию
        self.mu = tk.DoubleVar(value=10)
        self.beta = tk.DoubleVar(value=0.1)
        self.epsilon = tk.DoubleVar(value=0.1)
        self.learning_rate = tk.DoubleVar(value=0.01)
        self.max_iter_gd = tk.IntVar(value=100)
        self.x0_1 = tk.DoubleVar(value=0.0)
        self.x0_2 = tk.DoubleVar(value=0.0)
        
        # Данные для таблицы
        self.results = []
        self.export_btn = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Создание интерфейса из трёх частей"""
        # Основной контейнер
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Левая панель с параметрами
        left_frame = ttk.LabelFrame(main_paned, text="Параметры метода")
        main_paned.add(left_frame, weight=1)
        
        # Центральная панель с таблицей
        center_frame = ttk.LabelFrame(main_paned, text="Результаты итераций")
        main_paned.add(center_frame, weight=1)
        
        # Правая панель с графиком
        right_frame = ttk.LabelFrame(main_paned, text="График")
        main_paned.add(right_frame, weight=4)
        
        # === Заполнение левой панели ===
        left_canvas = tk.Canvas(left_frame)
        left_scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=left_canvas.yview)
        left_scrollable_frame = ttk.Frame(left_canvas)
        
        left_scrollable_frame.bind(
            "<Configure>",
            lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all"))
        )
        
        left_canvas.create_window((0, 0), window=left_scrollable_frame, anchor="nw")
        left_canvas.configure(yscrollcommand=left_scrollbar.set)
        
        left_canvas.pack(side="left", fill="both", expand=True)
        left_scrollbar.pack(side="right", fill="y")
        
        param_frame = ttk.Frame(left_scrollable_frame)
        param_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Информация о задаче
        ttk.Label(param_frame, text="ЗАДАЧА:", font=("Arial", 11, "bold")).grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky="w")
        
        # Целевая функция
        ttk.Label(param_frame, text="Целевая функция:", font=("Arial", 10, "bold")).grid(row=1, column=0, sticky="nw", pady=5)
        func_label = tk.Label(param_frame, text="f(x₁, x₂) = eˣ¹ - x₁·x₂ + x₂²", font=("Cambria", 14))
        func_label.grid(row=1, column=1, sticky="w", pady=5)
        
        # Ограничения
        ttk.Label(param_frame, text="Ограничения:", font=("Arial", 10, "bold")).grid(row=2, column=0, sticky="nw", pady=5)
        constraints_frame = ttk.Frame(param_frame)
        constraints_frame.grid(row=2, column=1, sticky="w", pady=5)
        
        g1_label = tk.Label(constraints_frame, text="g₁(x) = x₁² + x₂² - 4 ≤ 0", font=("Cambria", 14))
        g1_label.pack(anchor="w")
        
        g2_label = tk.Label(constraints_frame, text="g₂(x) = 2x₁ + x₂ - 2 ≤ 0", font=("Cambria", 14))
        g2_label.pack(anchor="w")
        
        # Разделитель
        ttk.Separator(param_frame, orient='horizontal').grid(row=3, column=0, columnspan=2, sticky="ew", pady=15)
        
        # Параметры решения
        ttk.Label(param_frame, text="ПАРАМЕТРЫ РЕШЕНИЯ:", font=("Arial", 11, "bold")).grid(row=4, column=0, columnspan=2, pady=(0, 10), sticky="w")
        
        # Начальная точка
        ttk.Label(param_frame, text="Начальная точка X₀:", font=("Arial", 10)).grid(row=5, column=0, sticky="w", pady=8)
        x0_frame = ttk.Frame(param_frame)
        x0_frame.grid(row=5, column=1, sticky="w", pady=8)
        ttk.Label(x0_frame, text="x₁ =", font=("Arial", 10)).pack(side=tk.LEFT)
        ttk.Entry(x0_frame, textvariable=self.x0_1, width=8, font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        ttk.Label(x0_frame, text="x₂ =", font=("Arial", 10)).pack(side=tk.LEFT, padx=(15, 0))
        ttk.Entry(x0_frame, textvariable=self.x0_2, width=8, font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        
        # Параметр μ
        ttk.Label(param_frame, text="Параметр μ:", font=("Arial", 10)).grid(row=6, column=0, sticky="w", pady=8)
        mu_entry = ttk.Entry(param_frame, textvariable=self.mu, width=10, font=("Arial", 10))
        mu_entry.grid(row=6, column=1, pady=8, sticky="w")
        
        # Параметр β
        ttk.Label(param_frame, text="Параметр β:", font=("Arial", 10)).grid(row=7, column=0, sticky="w", pady=8)
        beta_entry = ttk.Entry(param_frame, textvariable=self.beta, width=10, font=("Arial", 10))
        beta_entry.grid(row=7, column=1, pady=8, sticky="w")
        
        # ε
        ttk.Label(param_frame, text="Критерий остановки ε:", font=("Arial", 10)).grid(row=8, column=0, sticky="w", pady=8)
        eps_entry = ttk.Entry(param_frame, textvariable=self.epsilon, width=10, font=("Arial", 10))
        eps_entry.grid(row=8, column=1, pady=8, sticky="w")
        
        ttk.Separator(param_frame, orient='horizontal').grid(row=9, column=0, columnspan=2, sticky="ew", pady=15)
        
        ttk.Label(param_frame, text="ПАРАМЕТРЫ ГРАДИЕНТНОГО СПУСКА:", font=("Arial", 11, "bold")).grid(row=10, column=0, columnspan=2, pady=(0, 10), sticky="w")
        
        # Шаг
        ttk.Label(param_frame, text="Шаг (learning rate):", font=("Arial", 10)).grid(row=11, column=0, sticky="w", pady=8)
        lr_entry = ttk.Entry(param_frame, textvariable=self.learning_rate, width=10, font=("Arial", 10))
        lr_entry.grid(row=11, column=1, pady=8, sticky="w")
        
        # Макс. итераций
        ttk.Label(param_frame, text="Макс. итераций спуска:", font=("Arial", 10)).grid(row=12, column=0, sticky="w", pady=8)
        max_iter_entry = ttk.Entry(param_frame, textvariable=self.max_iter_gd, width=10, font=("Arial", 10))
        max_iter_entry.grid(row=12, column=1, pady=8, sticky="w")
        
        # Кнопки
        btn_frame = ttk.Frame(param_frame)
        btn_frame.grid(row=13, column=0, columnspan=2, pady=20)

        ttk.Button(btn_frame, text="ЗАПУСТИТЬ МЕТОД", command=self.run_barrier_method, width=25).pack(pady=2)

        ttk.Button(btn_frame, text="ЭКСПОРТ В CSV", command=self.export_to_csv, width=25, state='disabled').pack(pady=2)
        self.export_btn = btn_frame.winfo_children()[-1]  # Сохраняем ссылку на кнопку
                
        # === Центральная панель с таблицей ===
        table_container = ttk.Frame(center_frame)
        table_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Скроллбары
        scroll_y = ttk.Scrollbar(table_container)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

        scroll_x = ttk.Scrollbar(table_container, orient=tk.HORIZONTAL)
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
                
        self.tree = ttk.Treeview(table_container, columns=("k", "mu", "point", "f", "B", "Phi", "muB"), show="headings", yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        
        scroll_y.config(command=self.tree.yview)
        scroll_x.config(command=self.tree.xview)
        
        columns_def = [
            ("k", "K", 40),
            ("mu", "μₖ", 80),
            ("point", "Xμₖ = Xₖ₊₁", 140),
            ("f", "F(Xμₖ)", 70),
            ("B", "B(Xμₖ)", 90),
            ("Phi", "Ф(μₖ)", 80),
            ("muB", "μₖ·B(Xμₖ)", 70)
        ]
        
        for col, text, width in columns_def:
            self.tree.heading(col, text=text)
            self.tree.column(col, width=width, anchor="center")
        
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        # === Правая панель с графиком ===
        self.fig = Figure(figsize=(6, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    # === Функции задачи ===
    def objective(self, x):
        """Целевая функция F(x1, x2) = e^x1 - x1*x2 + x2^2"""
        return math.exp(x[0]) - x[0] * x[1] + x[1]**2
    
    def g1(self, x):
        """Ограничение 1: x1^2 + x2^2 - 4 ≤ 0"""
        return x[0]**2 + x[1]**2 - 4
    
    def g2(self, x):
        """Ограничение 2: 2*x1 + x2 - 2 ≤ 0"""
        return 2 * x[0] + x[1] - 2
    
    def is_feasible(self, x):
        """Проверка допустимости точки"""
        return self.g1(x) < 0 and self.g2(x) < 0
    
    def barrier(self, x):
        """Барьерная функция B(x) = -1/g1(x) - 1/g2(x)"""
        g1_val = self.g1(x)
        g2_val = self.g2(x)
        
        if g1_val >= 0 or g2_val >= 0:
            return float('inf')
        
        if abs(g1_val) < 1e-10 or abs(g2_val) < 1e-10:
            return float('inf')
        
        return -1.0 / g1_val - 1.0 / g2_val
    
    def auxiliary(self, x, mu):
        """Вспомогательная функция Φ(x, μ) = F(x) + μ·B(x)"""
        b = self.barrier(x)
        if b == float('inf'):
            return float('inf')
        return self.objective(x) + mu * b
    
    # === Градиенты ===
    def grad_f(self, x):
        """Градиент целевой функции"""
        df_dx1 = math.exp(x[0]) - x[1]
        df_dx2 = -x[0] + 2 * x[1]
        return np.array([df_dx1, df_dx2])
    
    def grad_g1(self, x):
        """Градиент первого ограничения"""
        return np.array([2 * x[0], 2 * x[1]])
    
    def grad_g2(self, x):
        """Градиент второго ограничения"""
        return np.array([2, 1])
    
    def grad_b(self, x):
        """Градиент барьерной функции"""
        g1_val = self.g1(x)
        g2_val = self.g2(x)
        
        if g1_val >= 0 or g2_val >= 0:
            return np.array([float('inf'), float('inf')])
        
        grad_g1 = self.grad_g1(x)
        grad_g2 = self.grad_g2(x)
        
        coeff1 = 1.0 / (g1_val * g1_val)
        coeff2 = 1.0 / (g2_val * g2_val)
        
        return coeff1 * grad_g1 + coeff2 * grad_g2
    
    def grad_phi(self, x, mu):
        """Градиент вспомогательной функции"""
        grad_f = self.grad_f(x)
        grad_b = self.grad_b(x)
        
        if np.any(np.isinf(grad_b)):
            return np.array([float('inf'), float('inf')])
        
        return grad_f + mu * grad_b
    
    # === Градиентный спуск ===
    def gradient_descent(self, x_start, mu, max_iter, learning_rate):
        """Минимизация вспомогательной функции градиентным спуском"""
        x = np.array(x_start, dtype=float)
        history = [x.copy()]
        lr = learning_rate
        
        for i in range(max_iter):
            grad = self.grad_phi(x, mu)
            
            if np.any(np.isinf(grad)) or np.any(np.isnan(grad)):
                break
            
            if np.linalg.norm(grad) < 1e-8:
                break
            
            x_new = x - lr * grad
            
            if self.is_feasible(x_new):
                x = x_new
                history.append(x.copy())
            else:
                lr *= 0.5
                if lr < 1e-10:
                    break
                continue
        
        return x, history
    
    # === Метод барьерных функций ===
    def run_barrier_method(self):
        """Запуск метода барьерных функций"""
        try:
            # Получение параметров
            mu = self.mu.get()
            beta = self.beta.get()
            epsilon = self.epsilon.get()
            learning_rate = self.learning_rate.get()
            max_iter_gd = self.max_iter_gd.get()
            x_start = np.array([self.x0_1.get(), self.x0_2.get()])
            
            # Проверка начальной точки
            if not self.is_feasible(x_start):
                messagebox.showerror("Ошибка", "Начальная точка недопустима!\nОна должна удовлетворять g₁(x) < 0 и g₂(x) < 0\n")
                return
            
            self.results = []
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            if self.export_btn:
                self.export_btn.config(state='disabled')
            
            x_current = x_start.copy()
            k = 1
            self.xk_points = [x_current.copy()]  # Только точки Xk
            
            # Основной цикл
            while True:
                # Минимизация вспомогательной функции
                x_opt, history = self.gradient_descent(x_current, mu, max_iter_gd, learning_rate)
                
                # Вычисление значений
                f_val = self.objective(x_opt)
                b_val = self.barrier(x_opt)
                phi_val = self.auxiliary(x_opt, mu)
                mu_b = mu * b_val
                
                # Сохранение результатов
                self.results.append({
                    'k': k,
                    'mu': mu,
                    'x': x_opt.copy(),
                    'f': f_val,
                    'b': b_val,
                    'phi': phi_val,
                    'mu_b': mu_b
                })
                
                # Добавляем точку Xk в список
                self.xk_points.append(x_opt.copy())
                
                # Форматирование значений для таблицы
                if mu == int(mu):
                    mu_str = f"{int(mu)}"
                else:
                    mu_str = f"{mu:.6f}".rstrip('0').rstrip('.')
                
                # Добавление в таблицу
                self.tree.insert("", tk.END, values=(
                    k,
                    mu_str,
                    f"({x_opt[0]:.4f}; {x_opt[1]:.4f})",
                    f"{f_val:.4f}",
                    f"{b_val:.4f}",
                    f"{phi_val:.4f}",
                    f"{mu_b:.4f}"
                ))
                
                # Прокрутка таблицы к последней записи
                self.tree.yview_moveto(1.0)
                self.tree.update_idletasks()
                
                # Проверка критерия остановки
                if mu_b < epsilon:
                    break
                
                # Обновление параметров
                mu *= beta
                x_current = x_opt.copy()
                k += 1
                
                if k > 50:
                    messagebox.showwarning("Предупреждение", "Достигнуто максимальное количество внешних итераций (50)")
                    break
            
            # Построение графика
            self.plot_results()
            
            # Активировать кнопку экспорта
            if self.export_btn:
                self.export_btn.config(state='normal')
                        
            messagebox.showinfo("Завершено", 
                              f"Метод барьерных функций завершён!\n"
                              f"Количество итераций: {k}\n"
                              f"Оптимальная точка: ({x_opt[0]:.6f}, {x_opt[1]:.6f})\n"
                              f"Значение функции: {f_val:.6f}")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка:\n{str(e)}")
    
    def plot_results(self):
        """Построение графика допустимой области и точек"""
        self.ax.clear()
        
        # Создание сетки для построения
        x1 = np.linspace(-3, 3, 400)
        x2 = np.linspace(-3, 3, 400)
        X1, X2 = np.meshgrid(x1, x2)
        
        # Границы ограничений
        g1_vals = X1**2 + X2**2 - 4
        g2_vals = 2*X1 + X2 - 2
        
        # Рисование допустимой области
        self.ax.contourf(X1, X2, np.maximum(g1_vals, g2_vals), levels=[-100, 0], colors=['lightgreen'], alpha=0.3)
        
        # Границы ограничений
        self.ax.contour(X1, X2, g1_vals, levels=[0], colors=['red'], linewidths=2)
        self.ax.contour(X1, X2, g2_vals, levels=[0], colors=['blue'], linewidths=2)
        
        # Целевая функция в виде контуров
        f_vals = np.exp(X1) - X1*X2 + X2**2
        self.ax.contour(X1, X2, f_vals, levels=20, colors='gray', alpha=0.5, linestyles='dashed')
        
        # Точки Xk (результаты внешних итераций) с соединением линиями
        if hasattr(self, 'xk_points') and len(self.xk_points) > 0:
            points = np.array(self.xk_points)
            
            # Рисуем линии, соединяющие Xk
            self.ax.plot(points[:, 0], points[:, 1], 'g-', linewidth=2.5, alpha=0.8, label='Траектория Xₖ')
            
            # Рисуем маркеры для каждой точки Xk
            for i, point in enumerate(points):
                if i == 0:
                    self.ax.plot(point[0], point[1], 'bo', markersize=10, label='X₀ (начальная)')
                elif i == len(points) - 1:
                    self.ax.plot(point[0], point[1], 'ro', markersize=12, label=f'X* (оптимальная)')
                else:
                    # Промежуточные точки Xk
                    self.ax.plot(point[0], point[1], 'ko', markersize=7)
            
            # Добавляем подписи над точками
            for i, point in enumerate(points):
                if i == 0:
                    # Подпись X₀ над точкой
                    self.ax.annotate('X₀', point, xytext=(0, 8), textcoords='offset points', fontsize=10, fontweight='bold', ha='center', color='blue')
                elif i == len(points) - 1:
                    self.ax.annotate('X*', point, xytext=(0, 8), textcoords='offset points', fontsize=10, fontweight='bold', ha='center', color='red')
                else:
                    self.ax.annotate(f'X_{i}', point, xytext=(0, 8), textcoords='offset points', fontsize=9, ha='center', color='black')
        
        # Настройка графика
        self.ax.set_xlabel('x₁', fontsize=11)
        self.ax.set_ylabel('x₂', fontsize=11)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='upper right', fontsize=9)
        self.ax.set_xlim(-3, 3)
        self.ax.set_ylim(-3, 3)
        
        self.canvas.draw()
        
    def export_to_csv(self):
        """Экспорт таблицы результатов в CSV файл"""
        if not self.results:
            messagebox.showwarning("Предупреждение", "Нет данных для экспорта. Сначала запустите метод.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"barrier_method_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        
        if filename:
            try:
                with open(filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
                    writer = csv.writer(csvfile, delimiter=';')
                    # Заголовки
                    writer.writerow(['K', 'μₖ', 'Xμₖ = Xₖ₊₁', 'F(Xμₖ)', 'B(Xμₖ)', 'Ф(μₖ)', 'μₖ·B(Xμₖ)'])
                    
                    # Данные
                    for result in self.results:
                        # Округляем μₖ до разумного количества знаков
                        mu_value = result['mu']
                        if abs(mu_value - round(mu_value, 6)) < 1e-10:
                            mu_str = str(round(mu_value, 6))
                        else:
                            mu_str = f"{mu_value:.6f}".rstrip('0').rstrip('.')
                        
                        writer.writerow([
                            result['k'],
                            mu_str,
                            f"({result['x'][0]:.6f}; {result['x'][1]:.6f})",
                            f"{result['f']:.6f}",
                            f"{result['b']:.6f}",
                            f"{result['phi']:.6f}",
                            f"{result['mu_b']:.6f}"
                        ])
                
                # Открытие файла в программе по умолчанию
                try:
                    if os.name == 'nt':  # Windows
                        os.startfile(filename)
                    elif os.name == 'posix':  # macOS и Linux
                        subprocess.run(['open', filename])  # macOS
                        # subprocess.run(['xdg-open', filename])  # Linux (раскомментируйте для Linux)
                except Exception as open_error:
                    messagebox.showwarning("Предупреждение", f"Файл сохранён, но не удалось его открыть:\n{filename}\n")
                
                messagebox.showinfo("Успех", f"Таблица успешно экспортирована в:\n{filename}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить файл:\n{str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = BarrierMethodApp(root)
    root.state('zoomed') 
    root.mainloop()