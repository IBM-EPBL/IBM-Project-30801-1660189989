# -*- coding: utf-8 -*-
"""DA_Assignment_3_Python.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1UnwWqw2PCFBDro1w0T__lvTKxX2dcsRL

## Exercises

Answer the questions or complete the tasks outlined in bold below, use the specific method described if applicable.

** What is 7 to the power of 4?**
"""

result = pow(7,4)
print(result)

"""** Split this string:**

    s = "Hi there Sam!"
    
**into a list. **
"""

s= "Hi there string!"
x = s.split()
print(x)





"""** Given the variables:**

    planet = "Earth"
    diameter = 12742

** Use .format() to print the following string: **

    The diameter of Earth is 12742 kilometers.
"""



planet = "Earth"
diameter = 12742
print('The diameter of {} is {} kilometers.'.format(planet,diameter));

"""** Given this nested list, use indexing to grab the word "hello" **"""

lst = [1,2,[3,4],[5,[100,200,['hello']],23,11],1,7]

lst= [1,2,[3,4],[5,[100,200,['hello']],23,11],1,7]
a=lst[3][1][2];
print(a)

"""** Given this nest dictionary grab the word "hello". Be prepared, this will be annoying/tricky **"""

d = {'k1':[1,2,3,{'tricky':['oh','man','inception',{'target':[1,2,3,'hello']}]}]}

d={'k1':[1,2,3,{'trichy':['oh','man','inception',{'target':[1,2,3,'hello']}]}]}
print(d['k1'][3]["trichy"][3]['target'][3])

"""** What is the main difference between a tuple and a list? **"""

tuple is immutable
tuples operations are Small
tuples consumes less memory

"""** Create a function that grabs the email website domain from a string in the form: **

    user@domain.com
    
**So for example, passing "user@domain.com" would return: domain.com**
"""

def domainGet(email):
 print("your domain is :"+email.split('@')[-1])
 email=input("please enter your email:>")
 domainGet(email)



"""** Create a basic function that returns True if the word 'dog' is contained in the input string. Don't worry about edge cases like a punctuation being attached to the word dog, but do account for capitalization. **"""



def findDog(st):
  if'dog' in st.lower():
    print("True")
  else:
    print("False")
st="Is there a dog here?"
findDog(st)

"""** Create a function that counts the number of times the word "dog" occurs in a string. Again ignore edge cases. **"""

value='this dog runs fasster than the other dog dude!';
def countdogs(value):
  count =0
  for word in value.lower().split():
    if word=='dog' or word=='dogs':
      count=count+1
      print(count)
countdogs(value)



"""### Problem
**You are driving a little too fast, and a police officer stops you. Write a function
  to return one of 3 possible results: "No ticket", "Small ticket", or "Big Ticket". 
  If your speed is 60 or less, the result is "No Ticket". If speed is between 61 
  and 80 inclusive, the result is "Small Ticket". If speed is 81 or more, the result is "Big    Ticket". Unless it is your birthday (encoded as a boolean value in the parameters of the function) -- on your birthday, your speed can be 5 higher in all 
  cases. **
"""

def caught_speeding(speed, is_birthday):
    
    if is_birthday:
        speeding = speed - 5
    else:
        speeding = speed
    
    if speeding > 80:
        return 'Big Ticket'
    elif speeding > 60:
        return 'Small Ticket'
    else:
        return 'No Ticket'

caught_speeding(81,False)

caught_speeding(81,True)

"""Create an employee list with basic salary values(at least 5 values for 5 employees)  and using a for loop retreive each employee salary and calculate total salary expenditure. """

def salary(hours_worked,wage):
 if hours_worked>40:
   return 40*wage+(hours_worked-40)
 else:
    return hours_worked*wage
hours_worked=50
wage=100
pay=salary(hours_worked,wage)
print(f"total salary:Rs.{pay:.2f}")

"""Create two dictionaries in Python:

First one to contain fields as Empid,  Empname,  Basicpay

Second dictionary to contain fields as DeptName,  DeptId.

Combine both dictionaries. 
"""

def Merge(dict1,dict2):
  res={**dict1,**dict2}
  return res
dict1={'Empid':1,'Empname': 'anu ','Basicpay':1000}
dict2={'DeptName': 'cse','Deptid':21}
dict3=Merge(dict1,dict2)
print(dict3)