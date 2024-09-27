# Dijkstra's Algorithm for Route Planning
![Shortest route visualisation using Djikstra's Algorithm](Data Structures and Algorithm/img.png)
## Overview

This is a project under CSC2103 Data Structures and Algorithm. This Python program helps users find the shortest path between two locations using **Dijkstra's Algorithm**. The program fetches data from **OpenStreetMap** and calculates optimal routes based on the user’s input preferences, including **mode of transportation** (driving, biking, or walking) and an **optimization factor** (time or distance). The final result includes the total travel distance, estimated travel time, and a visual map generated in HTML.

## Features

- **Real-world data from OpenStreetMap** to calculate routes.
- **Multi-modal transportation**: Choose between driving, walking, and biking.
- **Optimizable routes**: Prioritize travel time or travel distance.
- **Interactive HTML map** output displaying the shortest route.
- Uses **multi-threaded Dijkstra's Algorithm** to calculate the shortest path.
  
## Libraries Used

- **osmnx**: For downloading and working with OpenStreetMap data.
- **networkx**: For modeling street networks and calculating shortest routes.
- **geopy**: For converting addresses into geographical coordinates.
- **heapq**: For implementing Dijkstra’s Algorithm using a priority queue.

## Usage

For more usage instructions and output results, refer to 'report.pdf' in the repository.
