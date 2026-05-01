import tkinter as tk
from tkinter import ttk, messagebox
import math
import numpy as np
import csv
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class BarrierMethod:
    def __init__(self):
        self.solution_history = []
        self.trajectory_points = []
        self.final_solution = None
        self.current_mu = None
        self.iteration_count = 0
        
    def objective(self, x):
        """Целевая функция F(x1, x2) = e^x1 - x1*x2 + x2^2"""
        return math.exp(x[0]) - x[0] * x[1] + x[1]**2
    
    def g1(self, x):
        """Ограничение 1: x1^2 + x2^2 - 4 ≤ 0"""
        return x[0]**2 + x[1]**2 - 4
    
    def g2(self, x):
        """Ограничение 2: 2*x1 + x2 - 2 ≤ 0"""
        return 2 * x[0] + x[1] - 2
    
    def barrier(self, x):
        """Барьерная функция B(x) = -1/g1(x) - 1/g2(x)"""
        g1_val = self.g1(x)
        g2_val = self.g2(x)
        
        # Проверка строгой внутренности
        if g1_val >= 0 or g2_val >= 0:
            return float('inf')
        
        return -1.0 / g1_val - 1.0 / g2_val
    
    def auxiliary(self, x, mu):
        """Вспомогательная функция Φ(x, μ) = F(x) + μ·B(x)"""
        return self.objective(x) + mu * self.barrier(x)
    
    def grad_f(self, x):
        """Градиент целевой функции"""
        df_dx1 = math.exp(x[0]) - x[1]
        df_dx2 = -x[0] + 2 * x[1]
        return np.array([df_dx1, df_dx2])
    
    def grad_b(self, x):
        """Градиент барьерной функции"""
        g1_val = self.g1(x)
        g2_val = self.g2(x)
        
        db_dx1 = (1.0 / (g1_val * g1_val)) * (2 * x[0]) + (1.0 / (g2_val * g2_val)) * 2
        db_dx2 = (1.0 / (g1_val * g1_val)) * (2 * x[1]) + (1.0 / (g2_val * g2_val)) * 1
        
        return np.array([db_dx1, db_dx2])
    
    def grad_phi(self, x, mu):
        """Градиент вспомогательной функции"""
        return self.grad_f(x) + mu * self.grad_b(x)
    
    def is_feasible(self, x, tol=1e-8):
        """Проверка допустимости точки (строгая внутренность)"""
        return self.g1(x) < -tol and self.g2(x) < -tol
    
    def golden_section_search(self, x, direction, mu, a, b, eps=1e-6):
        """
        Метод золотого сечения для одномерной оптимизации вдоль направления
        Находит α, минимизирующее Φ(x + α·direction, μ)
        """
        phi = 0.618  # Коэффициент золотого сечения
        
        def objective(alpha):
            x_alpha = [x[0] + alpha * direction[0], x[1] + alpha * direction[1]]
            # Проверка допустимости
            if not self.is_feasible(x_alpha):
                return float('inf')
            return self.auxiliary(x_alpha, mu)
        
        # Начальный этап
        lamda = a + (1 - phi) * (b - a)
        mu_val = a + phi * (b - a)
        
        f_lamda = objective(lamda)
        f_mu = objective(mu_val)
        
        # Основной цикл
        while b - a > eps:
            if f_lamda > f_mu:
                a = lamda
                lamda = mu_val
                f_lamda = f_mu
                mu_val = a + phi * (b - a)
                f_mu = objective(mu_val)
            else:
                b = mu_val
                mu_val = lamda
                f_mu = f_lamda
                lamda = a + (1 - phi) * (b - a)
                f_lamda = objective(lamda)
        
        return (a + b) / 2
    
    def optimize_with_golden_section(self, x_start, mu, max_iter=50, tol=1e-6):
        """
        Безусловная оптимизация методом покоординатного спуска с одномерной оптимизацией золотым сечением
        """
        x = np.array(x_start, dtype=float)
        
        for iteration in range(max_iter):
            x_old = x.copy()
            
            # Спуск по x1
            direction1 = np.array([1.0, 0.0])
            alpha1 = self.golden_section_search(x, direction1, mu, -2.0, 2.0, tol)
            x = x + alpha1 * direction1
            
            # Спуск по x2
            direction2 = np.array([0.0, 1.0])
            alpha2 = self.golden_section_search(x, direction2, mu, -2.0, 2.0, tol)
            x = x + alpha2 * direction2
            
            # Проверка сходимости
            if np.linalg.norm(x - x_old) < tol:
                break       
        return x
    
    def run_algorithm(self, x_start, mu_1, beta, epsilon):
        """
        Алгоритм метода барьерных функций (строго по лекции)
        """
        self.solution_history.clear()
        self.trajectory_points.clear()
        self.trajectory_points.append([x_start[0], x_start[1], self.objective(x_start), 0])
        
        x_k = np.array(x_start, dtype=float)
        mu_k = mu_1
        k = 1
        table_data = []
        
        # Проверка начальной точки
        if not self.is_feasible(x_k):
            raise ValueError(f"Начальная точка ({x_k[0]:.4f}, {x_k[1]:.4f}) недопустима!")
        
        while True:
            # Шаг 1: Решение задачи безусловной оптимизации методом золотого сечения
            x_next = self.optimize_with_golden_section(x_k, mu_k)
            
            # Сохраняем точку для графика
            self.trajectory_points.append([
                x_next[0], x_next[1], 
                self.objective(x_next), 
                mu_k
            ])
            
            # Вычисляем значения
            F_val = self.objective(x_next)
            B_val = self.barrier(x_next)
            muB_val = mu_k * B_val
            Phi_val = self.auxiliary(x_next, mu_k)
            
            # Сохраняем историю
            self.solution_history.append({
                'k': k,
                'mu': mu_k,
                'x': x_next.copy(),
                'F': F_val,
                'B': B_val,
                'Phi': Phi_val,
                'muB': muB_val
            })
            
            # Добавляем строку в таблицу
            row = {
                'K': k,
                'μ_k': f"{mu_k:.4f}".replace('.', ','),
                'X_{k+1}=Xμ_k': f"({x_next[0]:.4f}, {x_next[1]:.4f})".replace('.', ','),
                'F(X_{k+1})': f"{F_val:.4f}".replace('.', ','),
                'B(Xμ_k)': f"{B_val:.4f}".replace('.', ','),
                'Θ(μ_k)': f"{Phi_val:.4f}".replace('.', ','),
                'μ_k·B(Xμ_k)': f"{muB_val:.4f}".replace('.', ',')
            }
            table_data.append(row)
            
            # Шаг 2: Проверка критерия остановки
            if muB_val < epsilon:
                self.final_solution = x_next
                self.iteration_count = k
                break
            
            # Защита от бесконечного цикла
            if mu_k < 1e-12 or k > 50:
                self.final_solution = x_next
                self.iteration_count = k
                break
            
            # Обновляем для следующей итерации
            mu_k = beta * mu_k
            x_k = x_next
            k += 1       
        return table_data

class BarrierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Метод барьерных функций - Вариант 8")
        self.root.geometry("1400x900")
        
        self.barrier = BarrierMethod()
        
        self.setup_ui()
        
    def setup_ui(self):
        # Создаем основной фрейм
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Создаем верхнюю панель (слева параметры, справа график)
        top_frame = ttk.Frame(main_frame)
        top_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Левая часть - параметры алгоритма
        input_frame = ttk.LabelFrame(top_frame, text="Параметры метода барьерных функций", padding="8")
        input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N), padx=(0, 10))
        
        # Информация о задаче
        info_text = """Задача (Вариант 8):
        MIN F(x₁, x₂) = e^x₁ - x₁·x₂ + x₂²
        
        Ограничения:
        g₁(x) = x₁² + x₂² - 4 ≤ 0
        g₂(x) = 2x₁ + x₂ - 2 ≤ 0"""
        info_label = tk.Label(input_frame, text=info_text, font=('Arial', 11), justify=tk.LEFT, fg="black")
        info_label.grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        
        ttk.Separator(input_frame, orient='horizontal').grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=3)
        
        # Начальная точка X1
        ttk.Label(input_frame, font=('Arial', 11), text="Начальная точка x₁:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=3)
        
        x1_frame = ttk.Frame(input_frame)
        x1_frame.grid(row=2, column=1, sticky=tk.W, padx=5, pady=3)
        
        ttk.Label(x1_frame, font=('Arial', 11), text="x₁ =").grid(row=0, column=0, padx=2)
        self.x1_1 = tk.StringVar(value="0.0")
        ttk.Entry(x1_frame, textvariable=self.x1_1, width=8).grid(row=0, column=1, padx=2)
        
        ttk.Label(x1_frame, font=('Arial', 11), text="x₂ =").grid(row=0, column=2, padx=2)
        self.x1_2 = tk.StringVar(value="0.0")
        ttk.Entry(x1_frame, textvariable=self.x1_2, width=8).grid(row=0, column=3, padx=2)
        
        # Параметр μ₁
        ttk.Label(input_frame, font=('Arial', 11), text="Начальный параметр μ₁:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=3)
        self.mu_1_var = tk.StringVar(value="10.0")
        ttk.Entry(input_frame, textvariable=self.mu_1_var, width=15).grid(row=3, column=1, sticky=tk.W, padx=5, pady=3)
        
        # Параметр β
        ttk.Label(input_frame, font=('Arial', 11), text="Параметр β (0 < β < 1):").grid(row=4, column=0, sticky=tk.W, padx=5, pady=3)
        self.beta_var = tk.StringVar(value="0.1")
        ttk.Entry(input_frame, textvariable=self.beta_var, width=15).grid(row=4, column=1, sticky=tk.W, padx=5, pady=3)
        
        # Точность ε
        ttk.Label(input_frame, font=('Arial', 11), text="Точность ε:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=3)
        self.eps_var = tk.StringVar(value="0.1")
        ttk.Entry(input_frame, textvariable=self.eps_var, width=15).grid(row=5, column=1, sticky=tk.W, padx=5, pady=3)
        
        # Фрейм для кнопок
        buttons_frame = ttk.Frame(input_frame)
        buttons_frame.grid(row=6, column=0, columnspan=2, pady=5)
        
        self.start_button = ttk.Button(buttons_frame, text="Запустить алгоритм", command=self.run_algorithm, width=20)
        self.start_button.grid(row=0, column=0, padx=5)
        
        self.save_button = ttk.Button(buttons_frame, text="Сохранить в CSV", command=self.save_to_csv, width=20)
        self.save_button.grid(row=0, column=1, padx=5)
        
        # Фрейм для вывода результата
        result_frame = ttk.LabelFrame(input_frame, text="Результат", padding="5")
        result_frame.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.result_label = tk.Label(result_frame, text="Алгоритм не запущен", font=('Arial', 8), justify=tk.LEFT)
        self.result_label.grid(row=0, column=0, padx=5, pady=2)
        
        # Правая часть - график
        chart_frame = ttk.LabelFrame(top_frame, text="График: линии уровня и траектория движения", padding="8")
        chart_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Создаем график с измененным размером для лучшего отображения
        self.fig = Figure(figsize=(8, 4.5), dpi=100, facecolor='white')
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Нижняя часть - таблица с прокруткой
        table_frame = ttk.LabelFrame(main_frame, text="Результаты вычислений по методу барьерных функций (формат из задания)", padding="10")
        table_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Создаем фрейм для таблицы с прокруткой
        table_container = ttk.Frame(table_frame)
        table_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Создаем таблицу
        columns = ('K', 'μ_k', 'X_{k+1}=Xμ_k', 'F(X_{k+1})', 'B(Xμ_k)', 'Θ(μ_k)', 'μ_k·B(Xμ_k)')
        self.tree = ttk.Treeview(table_container, columns=columns, show='headings')
        
        column_widths = {'K': 50, 'μ_k': 100, 'X_{k+1}=Xμ_k': 140, 'F(X_{k+1})': 100, 'B(Xμ_k)': 100, 'Θ(μ_k)': 100, 'μ_k·B(Xμ_k)': 100}
        
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=column_widths.get(col, 100), anchor='center')
        
        v_scrollbar = ttk.Scrollbar(table_container, orient=tk.VERTICAL, command=self.tree.yview)
        h_scrollbar = ttk.Scrollbar(table_container, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Настройка весов
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=2)
        top_frame.columnconfigure(0, weight=0)
        top_frame.columnconfigure(1, weight=1)
        top_frame.rowconfigure(0, weight=1)
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        table_container.columnconfigure(0, weight=1)
        table_container.rowconfigure(0, weight=1)
    
    def update_result_display(self):
        """Обновление отображения результата"""
        if self.barrier.final_solution is not None:
            f_opt = self.barrier.objective(self.barrier.final_solution)
            
            result_text = f"""Оптимальное решение: x* = ({self.barrier.final_solution[0]:.6f}, {self.barrier.final_solution[1]:.6f})
            Значение функции: F* = {f_opt:.5f}
            Итераций: {self.barrier.iteration_count}
            g₁(x*) = {self.barrier.g1(self.barrier.final_solution):.5f} ≤ 0
            g₂(x*) = {self.barrier.g2(self.barrier.final_solution):.5f} ≤ 0"""
            self.result_label.config(text=result_text, font=('Arial', 11), fg="black")
        else:
            self.result_label.config(text="Алгоритм не запущен", fg="gray")
    
    def plot_function_contour(self, ax, x_range, y_range):
        """Построение контурного графика функции с корректным масштабом"""
        x_range_adjusted = (x_range[0], x_range[1])
        y_range_adjusted = (y_range[0], y_range[1])
        
        x = np.linspace(x_range_adjusted[0], x_range_adjusted[1], 200)
        y = np.linspace(y_range_adjusted[0], y_range_adjusted[1], 200)
        X, Y = np.meshgrid(x, y)
        
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = self.barrier.objective([X[i, j], Y[i, j]])
        
        # Автоматические уровни
        z_min = np.min(Z)
        z_max = np.max(Z)
        levels = np.linspace(z_min, z_max, 30)
        contour = ax.contour(X, Y, Z, levels=levels, colors='black', linewidths=0.5, alpha=0.6)
        ax.clabel(contour, inline=True, fontsize=7, fmt='%.1f')
        
        # Ограничение 1: x₁² + x₂² = 4
        theta = np.linspace(0, 2*np.pi, 100)
        x_circle = 2 * np.cos(theta)
        y_circle = 2 * np.sin(theta)
        ax.plot(x_circle, y_circle, 'r-', linewidth=2, label='x₁² + x₂² = 4')
        
        # Ограничение 2: 2x₁ + x₂ = 2
        x_line = np.linspace(x_range_adjusted[0], x_range_adjusted[1], 100)
        y_line = 2 - 2 * x_line
        ax.plot(x_line, y_line, 'b-', linewidth=2, label='2x₁ + x₂ = 2')
        
        # Допустимая область
        mask_g1 = X**2 + Y**2 <= 4
        mask_g2 = 2*X + Y <= 2
        mask = mask_g1 & mask_g2
        ax.contourf(X, Y, mask, levels=[0.5, 1], colors='green', alpha=0.1)
        
        return X, Y, Z
    
    def update_chart(self):
        """Обновление графика с динамическим диапазоном, но всегда видна вся функция"""
        self.ax.clear()
        
        if not self.barrier.trajectory_points:
            return
        
        points = np.array([[p[0], p[1]] for p in self.barrier.trajectory_points])
        
        # ФИКСИРОВАННЫЙ ПОЛНЫЙ ДИАПАЗОН (чтобы всегда была видна вся допустимая область)
        x_range = (-6, 4)    # x1 от -6 до 4 (видна вся окружность)
        y_range = (-4, 4)    # x2 от -4 до 4
        
        # Строим контуры на ВСЁМ диапазоне
        self.plot_function_contour(self.ax, x_range, y_range)
        
        # Траектория
        self.ax.plot(points[:, 0], points[:, 1], 'k-o', linewidth=2, markersize=6, markerfacecolor='black', markeredgecolor='black', label='Траектория')
        
        # Номера точек
        for i, point in enumerate(self.barrier.trajectory_points):
            self.ax.annotate(f'{i}', (point[0], point[1]), xytext=(5, 5), textcoords='offset points', fontsize=8, color='black', fontweight='bold')
        
        # Начальная и конечная точки
        if len(points) > 0:
            self.ax.plot(points[0, 0], points[0, 1], 'go', markersize=10, markerfacecolor='green', markeredgecolor='black', label='Начальная точка')
            self.ax.plot(points[-1, 0], points[-1, 1], 'ro', markersize=10, markerfacecolor='red', markeredgecolor='black', label='Оптимальная точка')
        
        self.ax.set_xlabel('x₁', fontsize=10)
        self.ax.set_ylabel('x₂', fontsize=10)
        self.ax.set_title('Линии уровня функции и траектория движения', fontsize=11)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='upper right', fontsize=8)
        
        self.ax.set_xlim(x_range[0], x_range[1])
        self.ax.set_ylim(y_range[0], y_range[1])

        self.ax.set_aspect('auto')
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def save_to_csv(self):
        """Сохранение таблицы результатов в CSV"""
        try:
            if not self.tree.get_children():
                messagebox.showwarning("Предупреждение", "Нет данных для сохранения. Сначала запустите алгоритм.")
                return
            
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Сохранить таблицу как"
            )
            
            if not filename:
                return
            
            columns = self.tree['columns']
            
            with open(filename, 'w', newline='', encoding='utf-8-sig') as file:
                writer = csv.writer(file, delimiter=';')
                writer.writerow(columns)
                
                for item in self.tree.get_children():
                    values = self.tree.item(item)['values']
                    writer.writerow(values)
            
            messagebox.showinfo("Успех", f"Таблица сохранена в файл:\n{filename}")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при сохранении:\n{str(e)}")
    
    def run_algorithm(self):
        """Запуск алгоритма"""
        try:
            x_start = [
                float(self.x1_1.get().replace(',', '.')),
                float(self.x1_2.get().replace(',', '.'))
            ]
            
            mu_1 = float(self.mu_1_var.get().replace(',', '.'))
            beta = float(self.beta_var.get().replace(',', '.'))
            epsilon = float(self.eps_var.get().replace(',', '.'))
            
            if mu_1 <= 0:
                messagebox.showerror("Ошибка", "μ₁ должно быть > 0")
                return
            
            if beta <= 0 or beta >= 1:
                messagebox.showerror("Ошибка", "β должно быть в интервале (0, 1)")
                return
            
            if epsilon <= 0:
                messagebox.showerror("Ошибка", "ε должно быть > 0")
                return
            
            table_data = self.barrier.run_algorithm(x_start, mu_1, beta, epsilon)
            
            # Обновление таблицы
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            for row in table_data:
                values = [row['K'], row['μ_k'], row['X_{k+1}=Xμ_k'], 
                         row['F(X_{k+1})'], row['B(Xμ_k)'], row['Θ(μ_k)'], 
                         row['μ_k·B(Xμ_k)']]
                self.tree.insert('', tk.END, values=values)
            
            self.update_chart()
            self.update_result_display()
            
            messagebox.showinfo("Успех", "Алгоритм успешно завершил работу!")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка:\n{str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = BarrierGUI(root)
    root.mainloop()