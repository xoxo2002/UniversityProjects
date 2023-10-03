package keyboardjumping.model

import keyboardjumping.MainApp.{Height, Width}
import scalafx.scene.paint.Color
import scalafx.scene.shape.{Line, Rectangle}

class Player(private var _x: Double, private var _y: Double) {
  //image of player jumping
  val playerHeight = 101
  val playerWidth = 65
  val Speed = 6
  val Gravity = 0.6
  val playerRect: Rectangle = new Rectangle {
    width = playerWidth
    height = playerHeight
    fill = Color.Transparent
    stroke = Color.Red
  }
  playerRect.x = _x
  playerRect.y = _y
  //the velocity
  var dx: Double = Speed
  var dy: Double = Speed

  //getter
  def x: Double = _x
  def y: Double = _y

  //setter
  def x_=(newX: Double): Unit = {
    _x = newX
    playerRect.x = _x
  }

  def y_=(newY: Double): Unit = {
    _y = newY
    playerRect.y = _y
  }

  def moveUp(): Unit = {
    dy -= Gravity
    _y -= dy
    playerRect.y = _y
  }

  def moveLeft(): Unit = {
    _x -= dx
    playerRect.x = _x
  }

  def moveRight(): Unit = {
    _x += dx
    playerRect.x = _x
  }

  def getBottom(): Double = {
    _y + playerHeight
  }

  def checkBound(platform: Platform): Unit = {
    //if the character touches the sides
    if (_x < 0) _x = 0
    if (_x > Width - 65) _x = (Width - 65)
    //if it touches the ground bounce back
    if (getBottom > Height) dy = 15
    //if colliding with platform bounce back
    if (isCollidingWithPlatform(platform: Platform) && (_x + 160 > platform.x) && (_x < platform.x + 160) && (dy < 0)) {
      if (_y + 90 < platform.y) {
        dy = 15
      }
    }
  }

  def isCollidingWithPlatform(platform: Platform): Boolean = {
    //check if the player's bounding box (rectangle) intersects with the platform's bounding box (rectangle)
    playerRect.intersects(platform.platformRect.getBoundsInLocal)
  }

  def isCollidingWithLineThreshold(Threshold: Line): Boolean = {
    //check if player is intersecting with the threshold line
    playerRect.intersects(Threshold.getBoundsInLocal)
  }
}

