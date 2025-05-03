# Custom decorator to log method calls
def log_method_call(func):
    def wrapper(*args, **kwargs):
        print(f"[LOG] Calling method: {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

# Parent Class
class Person:
    def __init__(self, name, age):
        self._name = name
        self._age = age

    @property
    def name(self):
        return self._name

    @property
    def age(self):
        return self._age

    @log_method_call
    def introduce(self):
        print(f"My name is {self.name}, and I am {self.age} years old.")

    @staticmethod
    def is_adult(age):
        return age >= 18

# Child Class
class Employee(Person):
    def __init__(self, name, age, employee_id, department):
        super().__init__(name, age)
        self.employee_id = employee_id
        self.department = department

    @log_method_call
    def introduce(self):
        print(f"My name is {self.name}, I am {self.age} years old, "
              f"and I work in {self.department} department. My ID is {self.employee_id}.")

    @classmethod
    def from_string(cls, emp_str):
        name, age, emp_id, dept = emp_str.split("-")
        return cls(name, int(age), emp_id, dept)

# Using the classes
person = Person("Alice", 30)
employee = Employee("Bob", 28, "E123", "Data Science")

person.introduce()
employee.introduce()

# Use static method
print("Is 17 an adult?", Person.is_adult(17))

# Use class method
new_emp = Employee.from_string("John-32-E789-Finance")
new_emp.introduce()