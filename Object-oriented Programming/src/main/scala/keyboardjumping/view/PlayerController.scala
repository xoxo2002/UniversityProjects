package keyboardjumping.view

import keyboardjumping.MainApp.{gc, m, reachCoordinate}
import keyboardjumping.model.{Platform, Player}
import scalafx.scene.image.{ImageView}
import scala.collection.mutable.ListBuffer

class PlayerController(player: Player, playerImageView: ImageView,  platforms: ListBuffer[Platform]) {
  var targetY = 270

  def handleMovement(): Unit = {
    if (reachCoordinate == false) {
      if (m > 0 && platforms(m).x == platforms(m - 1).x) {
        player.moveUp()
        gc.isJumping = true
      }
      else if (platforms(m).x == 130 && platforms(m).x - 10 < player.x) {
        player.moveUp()
        player.moveLeft()
        gc.isJumping = true
      }
      else if (platforms(m).x == 360 && platforms(m).x + platforms(m).width / 2 + 10 > player.x) {
        player.moveUp()
        player.moveRight()
        gc.isJumping = true
      }
      if (player.isCollidingWithPlatform(platforms(m)) && (player.x + 160 > platforms(m).x) && (player.x < platforms(m).x + 160) && (player.dy < 0)) {
        if (player.y + 90 < platforms(m).y) {
          reachCoordinate = true
          m = m + 1
          gc.isJumping = false
        }
      }
    }
  }

  def checkPlatformCollision(): Unit = {
    for (platform <- platforms) {
      player.checkBound(platform)
    }
  }

  def handleThresholdCollision(): Unit = {
    //if over thresh then move along
    if (player.y < gc.ScrollThreshold) {
      player.y = gc.ScrollThreshold
    }
  }

  def updatePlayerImage(): Unit = {
    playerImageView.layoutX = player.x
    playerImageView.layoutY = player.y
    playerImageView.toFront()
  }
}
