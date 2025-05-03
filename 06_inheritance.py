
class Person:
    def __init__(self,name,age):
        self.name = name
        self.age = age 
    
    def introduce(self):
        print(f'my name is {self.name} and my age is {self.age}')
    

class Employee(Person):
    def __init__(self,name,age,employee_id):
        super().__init__(name,age)
        self.employee_id = employee_id
    
    def introduce(self):
        print(f'my name is {self.name} and my age is {self.age} and my employee id is {self.employee_id}')
    
        

person = Person('Alice',29)
employee = Employee('Bob',35,12345)

person.introduce()
employee.introduce()

