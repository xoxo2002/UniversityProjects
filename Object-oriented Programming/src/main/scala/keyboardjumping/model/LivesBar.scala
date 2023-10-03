package keyboardjumping.model

import keyboardjumping.MainApp

import scalafx.scene.image.{Image, ImageView}
import scalafx.scene.layout.{HBox}

//LiveBar object to contain lives
object LivesBar {
  private var NumLives = 3
  private val lifeImage = new Image("keyboardjumping/view/images/lives.png")

  val livesIndicator: HBox = new HBox {
    spacing = 5
    prefHeight = 30
    style = "-fx-padding: 10;"
  }

  //update LiveBar with the number of lives
  private def newLivesBar: Seq[ImageView] = Seq.fill(NumLives) {
      new ImageView() {
        //size of the picture
        image = lifeImage
        fitWidth = 20
        fitHeight = 25
      }
    }
  livesIndicator.children = newLivesBar

  //reset LivesBar back to 3 lives
  def resetLives(): Unit = {
    NumLives = 3
    livesIndicator.children = newLivesBar
  }

  def addLives(): Unit = {
    NumLives = NumLives + 1
    livesIndicator.children = newLivesBar
  }

  def deductLives(): Unit = {
    NumLives = NumLives - 1
    livesIndicator.children = newLivesBar
    if (isLivesBarEmpty){
      MainApp.showGameOverScreen()
    }
  }

  private def isLivesBarEmpty: Boolean = livesIndicator.children.isEmpty
}
