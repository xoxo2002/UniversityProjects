# Keyboard Typing Game

## Overview
This project is a part of my **Object-Oriented Programming (PRG2104)** coursework. It involves developing a GUI-based game, where the player controls a bunny that leaps onto platforms by typing words correctly. The goal of the game is to promote fast and accurate typing through engaging gameplay. The player must type the displayed words correctly to move forward, while managing a timer, score, and limited lives. This project was built using **Scala** and the **ScalaFX** library, incorporating **SceneBuilder** for UI design. To see a live demonstration of the game, head to this link: [Demonstration Video](https://youtu.be/HAya4sCpurE)

![Game Main Menu](/media/Picture 1.gif)

## Game Features
- **Main Menu**: The game starts here, offering options to view instructions or begin the game.
- **Game Display**: The player interacts with this interface, where words appear on platforms. The player must type the words correctly to advance.
- **Timer, Lives, and Score**: The game is time-limited. The player has three lives, and each incorrect word deducts one life. The score increments with each word completed.

## Object-Oriented Concepts
The game employs core object-oriented programming (OOP) principles:
- **Classes and Objects**: The code is structured using a **Model-View-Controller (MVC)** architecture. Each game element, such as the player, platforms, timer, and lives bar, is represented by its own class.
- **Encapsulation and Modularity**: Game logic is handled centrally by the `GameController`, while specific functions are managed by modular controller classes, such as `PlayerController` and `PlatformController`.
- **Abstraction**: The game simplifies complex actions into high-level methods, like `generatePlatform()` and `isCollidingWithPlatforms()`.
- **Inheritance and Polymorphism**: Though the current version has limited use of these concepts, the codebase is designed to support future development with more game elements, difficulty levels, and dynamic behavior through inheritance.

