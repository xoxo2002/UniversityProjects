# Daily Sales Summarization - Multi-threaded Java Application

## Overview

This project is an assignment for the course CSC2044 Concurrent Programming. The project is a Java-based program that processes daily sales data from different branches around the world. The program generates a daily sales report with the following outcomes:
1. Total units sold of each product.
2. Total daily profits from all branches.
3. The branch with the lowest daily profits.

The program uses **multi-threading** to improve performance, especially for large datasets with varying numbers of sales records. This allows parallel processing of data for faster calculations.

## Features

- **Multi-threading**: The calculations of total units sold, total daily profits, and identification of the branch with the lowest profits are all done using multiple threads.
- **Scalability**: The program can handle datasets with different numbers of sales records.
- **Modular**: The program is structured to make it easy to modify or extend for future features.
