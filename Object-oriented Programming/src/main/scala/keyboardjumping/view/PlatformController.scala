package keyboardjumping.view

import keyboardjumping.MainApp.{gc}
import keyboardjumping.model.{Platform, WordSelector}
import scalafx.geometry.Pos
import scalafx.scene.control.Label
import scalafx.scene.image.ImageView
import scalafx.scene.layout.{AnchorPane, HBox}
import scalafx.scene.text.Font
import scala.collection.mutable.ListBuffer
import scala.util.Random

class PlatformController(gameDisplayRoot: AnchorPane, platforms: ListBuffer[Platform], platformsImageView: ListBuffer[ImageView], wordsView: ListBuffer[HBox]) {
  // Generate platforms
  var scroll = 0.0
  private val MaxNoOfPlatforms =  20
  var currentPlatformX = 130
  var currentPlatformY = 270
  val wordSelector = new WordSelector

  def generatePlatform(): Unit = {
    for (i <- 0 to MaxNoOfPlatforms) {
      //create and display platform
      val platform = new Platform(currentPlatformX, currentPlatformY)
      platforms += platform
      val platformImageView = platform.getView()
      platformsImageView += platformImageView

      gameDisplayRoot.children.add(platformImageView)
      //gameDisplayRoot.children.add(platform.platformRect)

      //add words
      val word = wordSelector.getRandomWord()
      val letterBox = new HBox {
        layoutX = currentPlatformX
        layoutY = currentPlatformY - platform.height
        alignment = Pos.Center
        spacing = 2
        prefWidth = platform.width // Custom width
        children = word.map { char =>
          new Label(char.toString.toUpperCase()) {
            font = Font.font(24)
          }
        }
      }
      wordsView += letterBox
      gameDisplayRoot.children.add(letterBox)

      //generate next platform coordinates
      val valueList: List[Int] = List(130, 360)

      val randomIndex = Random.nextInt(valueList.length)
      val x = valueList(randomIndex)
      val y = -130 // Calculate the Y-coordinate of the platform based on your logic
      currentPlatformX = x
      currentPlatformY = currentPlatformY + y
    }
    wordSelector.resetWords
  }

  def updatePlatformsPosition(): Unit = {
    if (gc.player.isCollidingWithLineThreshold(gc.line)) {
      for (x <- 0 to platforms.length - 1) {
        scroll = gc.player.dy
        val newY = platforms(x).update(scroll)
        val newWordY = newY - 30

        platformsImageView(x).layoutY = newY
        wordsView(x).layoutY = newWordY
      }
    }
  }

}
