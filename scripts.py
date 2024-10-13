# birthday-cake-candles

import math
import os
import random
import re
import sys

#Birthday Cake Candles
def birthdayCakeCandles(candles):
    counter=0
    some=max(candles) 
    for x in candles:
        if x==some:
            counter+=1
    return counter

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()
# Number Line Jumps

import math
import os
import random
import re
import sys


def kangaroo(x1, v1, x2, v2):
    # Set the kangaroo with the smaller starting position to mini
    maxi = x2
    maxiv = v2
    mini = x1
    miniv = v1

    # Adjust if x2 starts behind x1
    if x2 < x1:
        mini = x2
        miniv = v2
        maxi = x1
        maxiv = v1

    # If they start at the same position, they meet
    if mini == maxi:
        return 'YES'

    # If the kangaroo behind has a slower or equal speed, no meeting
    if miniv <= maxiv:
        return 'NO'

    # Simulate jumps and check if they land on the same spot
    while mini < maxi:
        mini += miniv
        maxi += maxiv
        if mini == maxi:
            return 'YES'

    # Return 'NO'
    return 'NO'

if __name__ == '__main__':
    # Read input
    first_multiple_input = input().rstrip().split()
    x1 = int(first_multiple_input[0])
    v1 = int(first_multiple_input[1])
    x2 = int(first_multiple_input[2])
    v2 = int(first_multiple_input[3])

    # Output result
    result = kangaroo(x1, v1, x2, v2)
    print(result)

# Arithmetic Operators
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a+b)
    print(a-b)
    print(a*b)

# Python Division

if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a//b)
    print(a/b)

# Recursive Digit Sum

import math
import os
import random
import re
import sys
def rec_function(value):
    if len(value)==1:
        return int(value)
    else:
        tot=0
        for digit in value:
            tot+=int(digit)
        return rec_function(str(tot))


def superDigit(n, k):
    tot=0
    tot=0
    for digit in n:
        tot+=int(digit)
    tot*=k
    return rec_function(str(tot))

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    n = first_multiple_input[0]

    k = int(first_multiple_input[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()

# Loops

if __name__ == '__main__':
    n = int(input())
    for x in range(n):
        print(x*x)

# Python If-Else

#!/bin/python3

import math
import os
import random
import re
import sys



if __name__ == '__main__':
    n = int(input().strip())
    if n%2==1:
        print("Weird")
    elif n%2==0 and n>=2 and n<=5:
        print("Not Weird")
    elif n%2==0 and n>=6 and n<=20:
        print("Weird")
    elif n%2==0 and n>20:
        print("Not Weird")
        
# Print Function

if __name__ == '__main__':
    n = int(input())
    x=""
    for i in range(n):
        x+=str(i+1)
    print(x)

# List Comprehensions
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())    
    total_count=x*y*z
    
    
    arr=[]
    
    for f in range(x+1):
        for s in range(y+1):
            for th in range(z+1):
                if sum([f,s,th])!=n:
                    arr.append([f,s,th])
    print(arr)

# Find the Runner-Up Score!

if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
    res=list(arr)
    current_m=max(res)
    final_res=[x for x in res if x!=current_m]
    print(max(final_res))

# Finding the percentage

if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    arr=student_marks[query_name]
    check=sum(arr)/len(arr)
    formatted_number = f"{check:.2f}"
    print(formatted_number)

# sWAP cASE
def swap_case(s):
    res = ""
    for char in s:
        if char.isupper():
            res += char.lower()
        elif char.islower():
            res += char.upper()
        else:
            res += char
    return res

# String Split and Join
def split_and_join(line):
    data=line.split(" ")
    return "-".join(data)

# What's Your Name?

def print_full_name(first, last):
    print('Hello '+first+' '+last+ '! You just delved into python.')

# Find a string

def count_substring(string, sub_string):
    counter=0
    for i in range(len(string)):
        if string[i:i+len(sub_string)]==sub_string:
            counter+=1
    return counter
# String Validators

if __name__ == '__main__':
    s = input()
    if any(count.isalnum() for count in s):
        print(True)
    else:
        print(False)
    if any(count.isalpha() for count in s):
        print(True)
    else:
        print(False)
    if any(count.isdigit() for count in s):
        print(True)
    else:
        print(False)
    if any(count.islower() for count in s):
        print(True)
    else:
        print(False)
    if any(count.isupper() for count in s):
        print(True)
    else:
        print(False)

# Text Alignment

thickness = int(input())
c = 'H'


for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))

# Text Wrap
def wrap(string, max_width):
    result=""
    for i in range(0,len(string),max_width):
        result+=string[i:i+max_width]+'\n'
    result=result.rstrip()
    return result

# Introduction to Sets
def average(array):
    setter=set(array)
    cur=sum(setter)
    return cur/len(setter)

# Symmetric Difference

first_length=input()
first=input()
second_length=input()
second=input()

first=first.split(" ")
second=second.split(" ")
first=set(first)
second=set(second)

first_length=int(first_length)
second_length=int(second_length)
res=[]
for element in first:
    if element not in second:
        res.append(int(element))
for element in second:
    if element not in first:
        res.append(int(element))
res.sort()
for x in res:
    print(x)

#Set.add()

counter=input()
adder=set()
for i in range(int(counter)):
    country=input()
    adder.add(country)
print(len(adder))

# Set .discard(), .remove() & .pop()

n = int(input())
s = set(map(int, input().split()))
third=input()
for i in range(int(third)):
    x=input()
    if x.startswith("pop"):
        s.pop()
        
    if x.startswith('remove'):
        
        res=x.split(' ')
        remel=int(res[1])
        if remel in s:
            s.remove(remel)
    if x.startswith('discard'):
        res=x.split(' ')
        s.discard(int(res[1]))
print(sum(s))

# Set .union() Operation

first=input()
second=input()
third=input()
fourth=input()

first=int(first)
third=int(third)

second=second.split(' ')
fourth=fourth.split(' ')
total_counter=len(second)

for element in fourth:
    
    if element not in second:
        total_counter+=1
print(total_counter)

# Set .intersecion() Operation

first=input()
second=input()
third=input()
fourth=input()
first=int(first)
third=int(third)
second=second.split(' ')
fourth=fourth.split(' ')
second=set([int(x) for x in second if x!=' '])
fourth=set([int(x) for x in fourth if x!=' '])

data=second.intersection(fourth)
print(len(data))

# Set .difference() Operation
first=input()
second=input()
third=input()
fourth=input()
second=set(second.split(' '))
fourth=set(fourth.split(' '))

print(len(second.difference(fourth)))

# Set .symmetric_difference() Operation

first=input()
second=input()
third=input()
fourth=input()
second=set(second.split(' '))
fourth=set(fourth.split(' '))
print(len(second.symmetric_difference(fourth)))

# Set Mutations

first=input()
second=input()
third=input()
second=second.split(' ')
second=set([int(x) for x in second])

third=int(third)

for i in range(third):
    operation=input()
    
    next_l=input()
    next_l=next_l.split(' ')
    next_l=[int(x) for x in next_l]
    setter=set(next_l)
    if operation.startswith('intersection_update'):
        second.intersection_update(setter)
    if operation.startswith('update'):
        second.update(setter)
    if operation.startswith('symmetric_difference_update'):
        second.symmetric_difference_update(setter)
    if operation.startswith('difference_update'):
        second.difference_update(setter)
print(sum(second))

# The Captain's Room

first=input()
second=input()
second=second.split(' ')
first_set=set()
second_set=set()
for element in second:
    if element not in first_set:
        first_set.add(element)
    else:
        second_set.add(element)
first_set.difference_update(second_set)
print(list(first_set)[0])

# Check Subset
first=input()
first=int(first)
for i in range(first):
    test_1=input()
    test_2=input()
    test_3=input()
    test_4=input()
    test_2=set(test_2.split(' '))
    test_4=set(test_4.split(' '))
    print(test_2.issubset(test_4))

# Check Strict Superset
def some():
    A=input()
    A=A.split(' ')
    A=set(A)
    number=int(input())
    for i in range(number):
        dataset=input()
        dataset=set(dataset.split(' '))
        if '' in dataset:
            dataset.remove('')
        B=A.copy()
        B.difference_update(dataset)
        if not A.issuperset(dataset) or (A.issuperset(dataset) and len(B)==0):
            return False
       
    return True    

    
    
print(some())

# No Idea!

first=input()
second=input()
third=input()
fourth=input()
third=third.split(' ')
fourth=fourth.split(' ')
num_el=first.split(' ')
starter=int(num_el[0])
ender=int(num_el[1])
tot=0

second=second.split(' ')
checker=dict()
for element in second:
    if element not in checker:
        checker[element]=1
    else:
        checker[element]+=1
        
        
for i in range(ender):
    if fourth[i] in checker:
        tot-=checker[fourth[i]]
    if third[i] in checker:
        tot+=checker[third[i]]
print(tot)

# Lists
if __name__ == '__main__':
    N = int(input())
    arr=[]
    for i in range(N):
        instruction=input()
        if instruction.startswith('insert'):
            data=instruction.split(' ')
            arr.insert(int(data[1]),int(data[2]))
        if instruction.startswith('remove'):
            data=instruction.split(' ')
            if int(data[1]) in arr:
                arr.remove(int(data[1]))
        if instruction.startswith('print'):
            print(arr)
        if instruction.startswith('append'):
            data=instruction.split(' ')
            arr.append(int(data[1]))
        
        if instruction.startswith('pop'):
            arr.pop()
        
        if instruction.startswith('reverse'):
            arr.reverse()
        if instruction.startswith('sort'):
            arr.sort()
        
# Tuples
if __name__ == '__main__':
    n = int(input())
    lister=input()
    lister= lister.split(' ')
    some=[int(x) for x in lister]
    print(hash(tuple(some)))

# Capitalize!

def solve(s):
    res=s.split(' ')
    current=''
    for x in res:
        current+=x.capitalize()
        current+=' '
    current=current[:-1]
    return current

# Detect Floating Point Number

import re
pattern = r"^[+\-]?(\d*\.\d+|\d+\.\d*)$"
first=input()
first=int(first)
for i in range(first):
    current=input()
    if re.match(pattern, current):
        try:
            float_value = float(current)
            print(True)
        except ValueError:
            print(False)
    else:
        print(False)

# Map and Lambda Function

cube = lambda x:x* x* x # complete the lambda function 

def fibonacci(n):
    start=[0,1]
    if n==0:
        return []
    if n==1:
        return [0]
    if n==2:
        return [0,1]
    for i in range(2,n):
        start.append(start[i-1]+start[i-2])
    return start

# Exceptions

a=input()
for i in range(int(a)):
    line=input()
    spl=line.split(" ")
    first=spl[0]
    second=spl[1]
    try:
        something=int(first)//int(second)
        print(something)
    except Exception as e:
        print("Error Code:",e)

# Calendar Module
import calendar
insert=input()
data_i=insert.split(' ')

year = int(data_i[2])
month = int(data_i[0])
day = int(data_i[1])
weekday = calendar.weekday(year, month, day)
weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
print(weekdays[weekday].upper())

# Time Delta
import math
import os
import random
import re
import sys
from datetime import datetime

def time_delta(t1, t2):
    date_format = "%a %d %b %Y %H:%M:%S %z"
    date1 = datetime.strptime(t1, date_format)
    date2 = datetime.strptime(t2, date_format)

    time_difference = (date2 - date1).total_seconds()
    return str(abs(int(time_difference)))
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        t1 = input()

        t2 = input()

        delta = time_delta(t1, t2)

        fptr.write(delta + '\n')

    fptr.close()

# XML 1 - Find the Score
total_attributes=0
def chi(check):
    global total_attributes
    cur=[elem for elem in check.iter() if elem is not check]
    for ch in cur:
        attributes = ch.attrib
        total_attributes+=len(attributes)


def get_attr_number(node):
    tree = etree.ElementTree(etree.fromstring(xml))
    root = tree.getroot()
    feed_element = root
    child_count = len(list(feed_element))
    global total_attributes
    for child in root:
        attributes = child.attrib
        total_attributes += len(attributes)
        chi(child)
    return total_attributes+len(root.attrib)


# String Formatting
def print_formatted(number):
    maxlen=len(str(bin(number))[2:])
    maxlenf=len(str(number)[2:])
    intermlen=maxlen-len(str(bin(1))[2:])-1
    for i in range(1,number+1):
        octal=str(oct(i))[2:]
        hexn=str(hex(i))[2:]
        binn=str(bin(i))[2:]
        hexn=hexn.upper()
        form_str="{:>"+str(maxlen)+"}"
        print_octal=form_str.format(octal)
        print_hex=form_str.format(hexn)
        print_binn=form_str.format(binn)
        print_n=form_str.format(i)
        print(print_n,print_octal,print_hex,print_binn)
    
# Zipped

data= input()
arr=data.split(' ')
numlines=int(arr[1])
numin=int(arr[0])
total_l=[]
for x in range(numlines):
    currentline=input()
    arrs=currentline.split(' ')
    arrs=[float(x) for x in arrs]
    total_l.append(arrs)
    
result=list(zip(*total_l))

for element in result:
    print(round(sum(element)/len(element),1))

# Athlete Sort
import math
import os
import random
import re
import sys



if __name__ == '__main__':
    nm = input().split()

    n = int(nm[0])

    m = int(nm[1])

    arr = []

    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))

    k = int(input())
    sorted_data = sorted(arr, key=lambda x: x[k])
    
    for x in range(n):
        for i in range(m):
            if i!=m-1:
                print(sorted_data[x][i], end=' ')
            else:
                print(sorted_data[x][i], end='')
        print('\n', end='')

# ginortS
input_str=input()

lowercase_letters=[]
uppercase_letters=[]
evendigits=[]
odddigits=[]
for letter in input_str:
    if letter.islower():
        lowercase_letters.append(letter)
    if letter.isupper():
        uppercase_letters.append(letter)
    if letter.isdigit():
        curnum=int(letter)
        if curnum%2==0:
            evendigits.append(curnum)
        else:
            odddigits.append(curnum)


lowercase_letters.sort()
uppercase_letters.sort()
evendigits.sort()
odddigits.sort()
finalstr=''
for x in lowercase_letters:
    finalstr+=x
for y in uppercase_letters:
    finalstr+=y

for t in odddigits:
    finalstr+=str(t)
for z in evendigits:
    finalstr+=str(z)
print(finalstr)

# Standartize Mobile Number Using Decorators
def wrapper(f):
    def fun(l):
        # complete the function
        some=[]
        for element in l:
            if len(element)==10:
                some.append(element)
            if len(element)==11:
                some.append(element[1:])
            if len(element)==12:
                some.append(element[2:])
            if len(element)==13:
                some.append(element[3:])
        final=sorted(some)
        for element in final:
            print("+91"+" "+element[:5]+" "+element[5:])
    return fun
# Decorators 2 - Name Directory
def person_lister(f):
    def inner(people):
        for element in people:
            element[2]=int(element[2])
            if element[3]=='M':
                element[3]='Mr.'
            if element[3]=='F':
                element[3]='Ms.'
        res=sorted(people, key=lambda x: x[2])
        i=0
        for element in res:
            currstr=element[3]+" "+element[0]+" "+element[1]
            res[i]=currstr
            i+=1
        return res
        
    return inner

# Re.split()
regex_pattern = r"[.,](?=\d)"

# Re.findall() & Re.finditer()
input_str=input()
pattern = r'(?<=[qwrtypsdfghjklzxcvbnmQWRTYPSDFGHJKLZXCVBNM])[aeiouAEIOU]{2,}(?=[qwrtypsdfghjklzxcvbnmQWRTYPSDFGHJKLZXCVBNM])'
substrings = re.findall(pattern, input_str)
if len(substrings)==0:
    print(-1)
else:
    for element in substrings:
        print(element)

# Re.start() & Re.end()

import re

S = input()
pat=input()

pattern = r'(?=('+pat+'))'
length_pattern=len(pat)-1

matches = re.finditer(pattern, S)

indices = []

enter_loop=False
for match in matches:
        enter_loop=True
        start_index = match.start()
        end_index = start_index+length_pattern
        indices.append((start_index, end_index))
if enter_loop:
    for ind in indices:
            print(ind)
else:
    print((-1,-1))

# Validating phone numbers
import re
first_n=int(input())
for i in range(first_n):
    inputstr=input()
    pattern=r'^[789]\d{9}$'
    if re.match(pattern, inputstr):
        print('YES')
    else:
        print('NO')
# Validating Roman Numerals
regex_pattern = r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"

# Validating and Parsing Email Addresses
import re
input_num=int(input())
pattern = r'^[a-zA-Z][a-zA-Z0-9._-]*@[a-zA-Z]+\.[a-zA-Z]{1,3}$'
for i in range(input_num):
    
    inputstr=input()
    lister=inputstr.split(' ')
    emaila=lister[1]
    emaila=emaila[1:len(emaila)-1]
    
    if re.match(pattern, emaila):
        print(inputstr)

# Hex Color Code
import re
num_lines=int(input())

hex_color_pattern = r'#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?\b'

for i in range(num_lines):
    cur_line=input()
    
    if not cur_line.startswith('#'):
        hex_colors = re.findall(hex_color_pattern, cur_line)
        for element in hex_colors:
              print(element)

# Alphabet Rangoli
def interm(listt,size,final_size,isize):
    listt=listt[isize-size:]
    middle_line=""
    for i in range(size-1,0,-1):
        middle_line+=listt[i]+'-'
    final_line=middle_line+listt[0]+middle_line[::-1]
    difference=int((final_size-len(final_line))/2)
    final_line='-'*difference+final_line+'-'*difference
    return final_line


def print_rangoli(size):
    if size==1:
        print("a")
        return
    listt = [chr(v) for v in range(ord('a'), ord('a') + 26)]
    final_size=4* size-3
    line_f=""
    middle_line=''
    for i in range(size,0,-1):
        check_line=interm(listt,i,final_size,size)
        if i!=size:
            line_f+=check_line+'\n'
        else:
            middle_line=check_line
    print(line_f[::-1][1:])
    print(middle_line)
    print(line_f)

# Viral advertising
import math
import os
import random
import re
import sys

def viralAdvertising(n):
    x=5
    result=0
    for i in range(n):
        t=math.floor(x/2)
        result+=t
        x=t*3
    return result

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()

# Insertion Sort-Part 1

import math
import os
import random
import re
import sys

def insertionSort1(n, arr):
    value=arr[n-1]
    something=False
    for i in range(n-2,-1,-1):
        if arr[i]>value:
            arr[i+1]=arr[i]
        if arr[i]<value:
            arr[i+1]=value
            something=True
        cur_str=" ".join([str(item) for item in arr])
        print(cur_str)
        if something:
            break
        if not something and i==0:
            arr[0]=value
            cur_str=" ".join([str(item) for item in arr])
            print(cur_str)
if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)


# Insertion Sort - Part 2
import math
import os
import random
import re
import sys

def insertionSort2(n, arr):
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
        curr=" ".join([str(item) for item in arr])
        print(curr)

if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)

# Mutations
def mutate_string(string, position, character):
    result=""
    for i in range(len(string)):
        if position==i:
            result+=character
        else:
            result+=string[i]
    return result

# Designer Door Mat

line_dims=input()
res=line_dims.split(' ')
height=int(res[0])
width=int(res[1])
middle_line=(height-1)//2
middle_w=(width-1)//2
number_int=1
result_start=''
for i in range(height):
    reversed_str=""
    for j in range(width):

        if i==middle_line and j==middle_w-3:
            print('WELCOME',end='')
            continue
        if i==middle_line and j<middle_w-3 or j>middle_w+3 and middle_line==i:
            print('-',end='')
        if j>middle_w-1-3*i and j<=middle_w-1-3*i+3*number_int:
            continue
        if j==middle_w-1-3*i and i!=middle_line:

            for x in range(number_int):
                print('.|.',end='')
                result_start+='.|.'

            print(reversed_str,end='')
            result_start+=reversed_str

            if i==middle_line and j>middle_w-3 and j<middle_w+4:
                continue

        if j<middle_w-1-3*i:
            reversed_str+='-'
            result_start+='-'
            print('-',end='')

    if i<middle_line:
        number_int+=2
    if i>middle_line:
        number_int-=2

    if i<=middle_line:
        print('\n',end='')
        result_start+='\n'

result_start=result_start.strip()
print(result_start[::-1])


# Merge the Tools!
def remove_char(string):
    length=len(string)
    setter=set()
    final_res=""
    for i in range(length):
        if string[i] not in setter:
            final_res+=string[i]
            setter.add(string[i])
    return final_res
def merge_the_tools(string, k):
    length=len(string)
    step=k
    for i in range(0,length,step):
        curr=string[i:i+step]
        result=remove_char(curr)
        print(result)

# The Minion Game

def generate_consonant_substrings(input_str):
    length = len(input_str)
    res_c = 0
    res_v = 0

    for i in range(length):
        if input_str[i] in {'A','E', 'O', 'U', 'I'}:
            res_v += length - i
        else:
            res_c += length - i

    return res_c, res_v
def minion_game(string):
        res_c,res_v = generate_consonant_substrings(string)
        if res_c>res_v:
            print("Stuart",res_c)
        if res_c==res_v:
            print("Draw")
        if res_v>res_c:
            print("Kevin",res_v)

# collection.Counter()

num_shoes=int(input())
sizes=input()
sizes=sizes.split()
setter=dict()
total=0
for i in range(num_shoes):
    if sizes[i] not in setter:
        setter[sizes[i]]=1
    else:
        setter[sizes[i]]+=1
buyers=int(input())
for i in range(buyers):
    buyer_price=input()
    arr=buyer_price.split()
    size=arr[0]
    price=int(arr[1])
    if size in setter:
        total+=price
        if setter[size]>1:
            setter[size]-=1
        else:
            del setter[size]
print(total)

# Arrays

def arrays(arr):
    arr=arr[::-1]
    a=numpy.array(arr,float)
    return a

# Transpose and Flatten

import numpy


first=input()
first=first.split()
lines_num=int(first[0])
col_num=int(first[1])
final=[]
for i in range(lines_num):
    string=input()
    arr=string.split()
    arr=[int(x) for x in arr]
    final.append(arr)
numpy_arr=numpy.array(final)
print(numpy.transpose(numpy_arr))
print(numpy_arr.flatten())

# Concatenate

import numpy as np

first = input()
curr_input = first.split()
first_iter = int(curr_input[0])
second_iter = int(curr_input[1])
num_elem = int(curr_input[2])
total_arrays = first_iter + second_iter
final_arr = np.array([], int)

for i in range(total_arrays):
    curr_str = input()
    curr_str = curr_str.split()
    curr_str = [int(x) for x in curr_str]
    curr_arr = np.array(curr_str, int)
    final_arr = np.concatenate((final_arr, curr_arr), axis=0)

final_arr = final_arr.reshape((-1, num_elem))
print(final_arr)

# Zeros and Ones
import numpy
insert_line=input()
input_data=insert_line.split()
final_arr=[]
for i in range(1,len(input_data)):
        final_arr.append(int(input_data[i]))


if len(input_data)>2:
    
    final_arr=tuple(final_arr)
    
    third=int(input_data[0])
    first_array=[]
    for i in range(third):
        x=numpy.zeros(final_arr, int)
        first_array.append(x)
    first_array=numpy.array(first_array)
    print(first_array)
    final_array=[]
    for j in range(third):
        x=numpy.ones(final_arr, int)
        final_array.append(x)
    final_array=numpy.array(final_array)
    print(final_array)

else:
    first=int(input_data[0])
    second=int(input_data[1])
    x=numpy.zeros((first,second), int)
    print(x)
    y=numpy.ones((first, second),int)
    print(y)

# DefaultDict Tutorial
from collections import defaultdict
n, m = map(int, input().split())

positions = defaultdict(list)


for i in range(1, n + 1):
    word = input()
    positions[word].append(i)


for i in range(1, m + 1):
    word = input()
    if word in positions:
        print(" ".join(map(str, positions[word])))
    else:
        print("-1")

# Collections.OrderedDict()
from collections import OrderedDict

N = int(input().strip())


item_prices = OrderedDict()

for _ in range(N):
    item_info = input().split()
    item_name = " ".join(item_info[:-1])
    item_price = int(item_info[-1])
    
    if item_name in item_prices:
        item_prices[item_name] += item_price
    else:
        item_prices[item_name] = item_price

for item_name, net_price in item_prices.items():
    print(item_name, net_price)

# Collections.namedtuple()
n = int(input().strip())

total_marks = 0
student_count = 0


column_names = input().split()
column_positions = {name: i for i, name in enumerate(column_names)}

for _ in range(n):
    data = input().split()
    
    try:
        marks = int(data[column_positions["MARKS"]])
        total_marks += marks
        student_count += 1
    except ValueError:
        pass

if student_count > 0:
    average_marks = total_marks / student_count
else:
    average_marks = 0

print("{:.2f}".format(average_marks))

# Collections.deque()

from collections import deque


d = deque()


N = int(input().strip())

for _ in range(N):
    operation = input().strip().split()
    if operation[0] == 'append':
        d.append(int(operation[1]))
    elif operation[0] == 'appendleft':
        d.appendleft(int(operation[1]))
    elif operation[0] == 'pop':
        d.pop()
    elif operation[0] == 'popleft':
        d.popleft()

print(" ".join(map(str, d)))


# Company Logo

import math
import os
import random
import re
import sys
from collections import Counter

def calcolare(s):

    char_count = Counter(s)
    most_common = char_count.most_common()
    most_common.sort(key=lambda x: (-x[1], x[0]))
    i=0
    for char, count in most_common:
        if i==3:
            break
        print(char, count)
        i+=1



if __name__ == '__main__':
    s = input()
    calcolare(s)

# Word Order
n = int(input().strip())


word_counts = {}
order_of_appearance = []

for _ in range(n):
    word = input().strip()
    if word not in word_counts:
        word_counts[word] = 1
        order_of_appearance.append(word)
    else:
        word_counts[word] += 1


num_distinct_words = len(order_of_appearance)


print(num_distinct_words)

for word in order_of_appearance:
    print(word_counts[word], end=" ")

# Piling Up!
def can_stack_cubes(n, blocks):
    left = 0
    right = n - 1
    prev_cube = float('inf')

    while left <= right:
        if blocks[left] >= blocks[right] and blocks[left] <= prev_cube:
            prev_cube = blocks[left]
            left += 1
        elif blocks[left] < blocks[right] and blocks[right] <= prev_cube:
            prev_cube = blocks[right]
            right -= 1
        else:
            return "No"

    return "Yes"

T = int(input().strip())

for _ in range(T):
    n = int(input().strip())
    blocks = list(map(int, input().strip().split()))
    result = can_stack_cubes(n, blocks)
    print(result)

# Group(), Groups() & Groupdict()
import re
S = input()
match = re.search(r'([a-zA-Z0-9])\1', S)

if match:
    print(match.group(1))
else:
    print(-1)

# Validating UID

import re

def is_valid_uid(uid):
    if len(uid) != 10:
        return False

    if len(re.findall(r'[A-Z]', uid)) < 2:
        return False

    if len(re.findall(r'\d', uid)) < 3:
        return False
    if not uid.isalnum():
        return False
    if len(set(uid)) != len(uid):
        return False

    return True

T = int(input())

for _ in range(T):
    uid = input()
    if is_valid_uid(uid):
        print("Valid")
    else:
        print("Invalid")

# Validating Credit Card Numbers

import re

pattern = r'^(?!.*(\d)(-?\1){3,})([456]\d{3}-?\d{4}-?\d{4}-?\d{4}|[456]\d{15})$'

N = int(input())
credit_cards = [input() for _ in range(N)]

for card in credit_cards:
    if re.match(pattern, card):
        print("Valid")
    else:
        print("Invalid")

# Validating Postal Codes
import re

# Define a regular expression pattern for valid credit card numbers
pattern = r'^(?!.*(\d)(-?\1){3,})([456]\d{3}-?\d{4}-?\d{4}-?\d{4}|[456]\d{15})$'

# Read input
N = int(input())
credit_cards = [input() for _ in range(N)]

# Check the validity of each credit card number
for card in credit_cards:
    if re.match(pattern, card):
        print("Valid")
    else:
        print("Invalid")
# Matrix Script

import re


N, M = map(int, input().split())

matrix = [input() for _ in range(N)]

decoded_script = ''.join(matrix[j][i] for i in range(M) for j in range(N))
decoded_script = re.sub(r'(?<=[a-zA-Z0-9])[^a-zA-Z0-9]+(?=[a-zA-Z0-9])', ' ', decoded_script)

print(decoded_script)

# Regex Substitution

import re

def modify_text(match):
    if match.group(0) == "&&":
        return "and"
    elif match.group(0) == "||":
        return "or"

N = int(input())

text = [input() for _ in range(N)]
modified_text = [re.sub(r'(?<= )(\&\&|\|\|)(?= )', modify_text, line) for line in text]

for line in modified_text:
    print(line)

# HTML Parser - Part 1
from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print("Start :", tag)
        for attr in attrs:
            print(f"-> {attr[0]} > {attr[1]}")

    def handle_endtag(self, tag):
        print("End   :", tag)

    def handle_startendtag(self, tag, attrs):
        print("Empty :", tag)
        for attr in attrs:
            print(f"-> {attr[0]} > {attr[1]}")        
        
        
start=int(input())
final=""
for i in range(start):
    final+=input()
    
parser = MyHTMLParser()

# HTML Parser - Part 2

from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_comment = False
        self.comment_type = None

    def handle_comment(self, data):
        if '\n' in data:
            self.comment_type = "Multi-line Comment"
        else:
            self.comment_type = "Single-line Comment"

        if self.comment_type == "Multi-line Comment":
            print(">>>", self.comment_type)
            print(data)
        elif self.comment_type == "Single-line Comment":
            print(">>>", self.comment_type)
            print(data)

    def handle_data(self, data):
        if data.strip():
            print(">>> Data")
            print(data)

if __name__ == "__main__":
    N = int(input())
    html_code = ""
    for _ in range(N):
        html_code += input() + '\n'

    parser = MyHTMLParser()
    parser.feed(html_code)

# Eye and Identity

import numpy
numpy.set_printoptions(legacy='1.13')

curr=input()
result=curr.split()
N=int(result[0])
M=int(result[1])

print(numpy.eye(N,M,k=0))

# Array Mathematics

import numpy as np
n, m = map(int, input().split())
arr_a = np.array([list(map(int, input().split())) for _ in range(n)])
arr_b = np.array([list(map(int, input().split())) for _ in range(n)])

addition = np.add(arr_a, arr_b)
subtraction = np.subtract(arr_a, arr_b)
multiplication = np.multiply(arr_a, arr_b)
division = np.floor_divide(arr_a, arr_b)
modulus = np.mod(arr_a, arr_b)
power = np.power(arr_a, arr_b)

print(addition)
print(subtraction)
print(multiplication)
print(division)
print(modulus)
print(power)

# Shape and Reshape
import numpy

curr=input()
curr=curr.split()
curr=[int(x) for x in curr]

my_array = numpy.array(curr)
print(numpy.reshape(my_array,(3,3)))

# Sum and Prod

import numpy
first=input()
first=first.split()
N=int(first[0])
M=int(first[1])
final=[]
for x in range(N):
    curr=input().split()
    curr=[int(x) for x in curr]
    final.append(curr)
my_array= numpy.array(final)
my_array=numpy.sum(my_array,axis=0)
print(numpy.prod(my_array))

# Min and Max

import numpy


final_arr=[]
starter=input().split()
N=int(starter[0])
M=int(starter[1])
for x in range(N):
    curr=input().split()
    curr=[int(x) for x in curr]
    final_arr.append(curr)
    
my_arr=numpy.array(final_arr)
arr=numpy.min(my_arr,axis=1)
print(numpy.max(arr))

# Mean, Var, and Std

import numpy
first= input().split()
N=int(first[0])
M=int(first[1])
final=[]
for x in range(N):
    curr= input().split()
    curr=[int(x) for x in curr]
    final.append(curr)
my_array = numpy.array(final)
print(numpy.mean(my_array, axis = 1))
print(numpy.var(my_array, axis = 0))
original_number=numpy.std(my_array)
rounded_number = round(original_number, 11)
print(rounded_number)

# Floor, Ceil and Rint

import numpy
numpy.set_printoptions(legacy='1.13')
first=input()
first=first.split()
final=[float(x) for x in first]

final=numpy.array(final)
print(numpy.floor(final))
print(numpy.ceil(final))
print(numpy.rint(final))

# Dot and Cross

import numpy as np

N = int(input())

A = []
for _ in range(N):
    row = list(map(int, input().split()))
    A.append(row)

B = []
for _ in range(N):
    row = list(map(int, input().split()))
    B.append(row)

A = np.array(A)
B = np.array(B)

result = np.dot(A, B)

print(result)

# Polynomials

import numpy as np

coefficients = list(map(float, input().split()))

x = float(input())

result = np.polyval(coefficients, x)

print(result)

# Linear Algebra

import numpy as np

N = int(input())

A = []
for _ in range(N):
    row = list(map(float, input().split()))
    A.append(row)

A = np.array(A)

determinant = np.linalg.det(A)

rounded_determinant = round(determinant, 2)

print(rounded_determinant)


# Inner and Outer

import numpy


import numpy as np


A = np.array(input().split(), dtype=int)
B = np.array(input().split(), dtype=int)

inner_product = np.inner(A, B)
print(inner_product)

outer_product = np.outer(A, B)
print(outer_product)

# Detect HTML Tags, Attributes and Attribute Values

from html.parser import HTMLParser
import math
import os
import random
import re
import sys

# Define a class to handle parsing of HTML tags and attributes
class MyHTMLParser(HTMLParser):

    # Handle opening tags and print
    def handle_starttag(self, tag, attrs):
        print(tag)
        if attrs:
            for attr in attrs:
                print(f"-> {attr[0]} > {attr[1]}")

  

    # Handle tags and print the tag name and its attributes
    def handle_startendtag(self, tag, attrs):
        print(tag)
        if attrs:
            for attr in attrs:
                print(f"-> {attr[0]} > {attr[1]}")

# get lines in html
n = int(input())
html_code = ""
for _ in range(n):
    html_code += input()

# Create MyHTMLParser and parse the collected HTML
parser = MyHTMLParser()
parser.feed(html_code)


# Nested Lists


if __name__ == '__main__':
    
    # Create an empty dictionary to store names and scores
    main = dict()
    
    # Loop to get input from the user
    for _ in range(int(input())):
        name = input()  # Get the name
        score = float(input())  # Get the score
        
        # Save the name and score in the dictionary
        main[name] = score

    # Get all the scores from the dictionary
    values = list(main.values())
    
    # Find the lowest score
    min_value = min(values)
    
    # Remove the lowest score from the list of scores
    values = [x for x in values if x != min_value]
    
    # Find the second lowest score
    second_min_value = min(values)
    
    # Create a list to store names with the second lowest score
    lister = []
    
    # Add names with the second lowest score to the list
    for x in main:
        if main[x] == second_min_value:
            lister.append(x)
    
    # Sort the list of names alphabetically
    lister = sorted(lister)
    
    # Print the names one by one
    for value in lister:
        print(value)



# XML2 - Find the Maximum Depth

# maxdepth to keep track of the deepest level found.
maxdepth = 0

# Define a recursive function 'depth' to calculate the depth
def depth(elem, level):
    global maxdepth  # maxdepth to track depth
    
    # If -1 , set it to 0.
    if level == -1:
        level = 0
    else:
        level += 1  # Increment the current level

    # check if level is bigger than maxdepth
    if level > maxdepth:
        maxdepth = level

    # Recursion
    for child in elem:
        depth(child, level)  # The depth of each child is calculated based on the current level.
