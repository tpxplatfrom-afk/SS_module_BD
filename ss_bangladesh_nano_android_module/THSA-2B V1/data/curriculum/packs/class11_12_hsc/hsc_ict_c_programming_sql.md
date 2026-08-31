# এইচএসসি (HSC) আইসিটি — সি প্রোগ্রামিং ভাষা ও ডাটাবেজ এসকিউএল (C & SQL)

## ১. সি প্রোগ্রামিং ভাষা (C Programming Language)
- ডেটা টাইপ ও সাইজ: int (2/4 bytes, %d), float (4 bytes, %f), double (8 bytes, %lf), char (1 byte, %c)।
- অপারেটর: এরিথমেটিক (+, -, *, /, %), রিলেশনাল (==, !=, >, <, >=, <=), লজিক্যাল (&&, ||, !)।
- কন্ট্রোল স্ট্রাকচার:
  * if-else শর্তাবলী এবং switch-case মাল্টিপল সিলেকশন।
  * লুপ (Loops): for loop, while loop, do-while loop।
- ফাংশন ও রিকার্সন (Functions & Recursion): ইউজার ডিফাইনড ফাংশন এবং রিকার্সিভ ফ্যাক্টরিয়াল / ফিবোনাচ্চি।
- অ্যারে (Array): এক মাত্রিক অ্যারে (1D) ও দুই মাত্রিক ম্যাট্রিক্স অ্যারে (2D)।
- আদর্শ সি প্রোগ্রাম উদাহরণ (১ থেকে N পর্যন্ত যোগফল):
```c
#include <stdio.h>
int main() {
    int n, sum = 0;
    printf("Enter n: ");
    scanf("%d", &n);
    for(int i = 1; i <= n; i++) {
        sum += i;
    }
    printf("Sum = %d\n", sum);
    return 0;
}
```

## ২. রিলেশনাল ডাটাবেজ ও এসকিউএল (RDBMS & SQL)
- ডিডিএল (DDL - Data Definition Language): CREATE TABLE, ALTER TABLE, DROP TABLE।
- ডিএমএল (DML - Data Manipulation Language): SELECT, INSERT, UPDATE, DELETE।
- প্রাইমারি কি (Primary Key) ও ফরেন কি (Foreign Key - দুটি টেবিলের মধ্যে রিলেশন তৈরি)।
- এসকিউএল কোয়েরি উদাহরণ:
  * ডেটা খোঁজা: `SELECT Name, GPA FROM Students WHERE GPA >= 5.0 ORDER BY Name ASC;`
  * নতুন ডেটা যুক্ত করা: `INSERT INTO Students (ID, Name, GPA) VALUES (101, 'Rahim', 5.0);`
  * ডেটা আপডেট: `UPDATE Students SET GPA = 4.8 WHERE ID = 101;`
  * জয়েন কোয়েরি: `SELECT A.Name, B.Department FROM Students A INNER JOIN Dept B ON A.DeptID = B.DeptID;`
