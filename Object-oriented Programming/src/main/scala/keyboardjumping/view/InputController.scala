package keyboardjumping.controller

import javafx.animation.TranslateTransition
import keyboardjumping.MainApp.{gameDisplayRoot, i, j, gc, reachCoordinate}
import scalafx.scene.input.{KeyCode, KeyEvent}
import scalafx.scene.layout.HBox
import scalafx.scene.paint.Color
import javafx.util.Duration
import keyboardjumping.MainApp
import keyboardjumping.model.LivesBar
import scala.collection.convert.ImplicitConversions.`list asScalaBuffer`
import scala.collection.mutable.ListBuffer


class InputController(wordsView: ListBuffer[HBox]) {
  private var upPressed = false
  private var downPressed = false
  private var leftPressed = false
  private var rightPressed = false
  private var spacePressed = false
  var lastCheckedChar = "-"
  val tremblingDuration = Duration.millis(50) // Duration for the trembling animation
  val tremblingDisplacement = 2.0 // Displacement range for trembling

  def onKeyPressed(event: KeyEvent): Unit = {
    event.code match {
      case KeyCode.Up => upPressed = true // handle up key press
      case KeyCode.Down => downPressed = true // handle down key press
      case KeyCode.Right => rightPressed = true // handle right key press
      case KeyCode.Left => leftPressed = true // handle left key press
      case KeyCode.Space => spacePressed = true // handle space key press
      case _ => // handle other keys if necessary
    }
  }

  def onKeyReleased(event: KeyEvent): Unit = {
    event.code match {
      case KeyCode.Up => upPressed = false
      case KeyCode.Down => downPressed = false
      case KeyCode.Right => rightPressed = false
      case KeyCode.Left => leftPressed = false
      //when user type in input
      case _ if event.code.isLetterKey =>
        //player presses key
        val pressedLetter = event.text.toString
        val word = wordsView(j).children.get(i).asInstanceOf[javafx.scene.control.Label]
        //if it is a match
        if (word.getText.equalsIgnoreCase(pressedLetter)) {
          word.setTextFill(Color.Green)
          //wordsView(j).children(i).textFill = Color.Green
          //move onto next children
          i = i + 1
        }
        else {
          //only one fault is counted for each character
          word.setTextFill(Color.Red)
          if (lastCheckedChar.equalsIgnoreCase(word.getText) == false){ //if they are not the same
            // print("same")
            gc.totalFault = gc.totalFault + 1
            word.setTextFill(Color.Red)
            LivesBar.deductLives()
          }
          val tremblingAnimation = new TranslateTransition(tremblingDuration, gameDisplayRoot.asInstanceOf[javafx.scene.Node])
          tremblingAnimation.setByX(tremblingDisplacement)
          tremblingAnimation.setByY(tremblingDisplacement)
          tremblingAnimation.setCycleCount(4)
          tremblingAnimation.setAutoReverse(true)
          tremblingAnimation.play()
          tremblingAnimation.setOnFinished(_ => {
            gameDisplayRoot.setTranslateX(0)
            gameDisplayRoot.setTranslateY(0)
          })
          lastCheckedChar = word.getText
        }

        if (i == wordsView(j).children.length) {
          i = 0
          wordsView(j).visible = false
          j = j + 1
          gc.totalScoreSoFar = gc.totalScoreSoFar + 10
          gc.timer.resetTimer
          reachCoordinate = false
          //print("total score:" + totalScoreSoFar)
          val totalScoreSoFar = gc.totalScoreSoFar
          gc.scoreLabel.text = s"Score: $totalScoreSoFar"
          //update view
          if(j == wordsView.length){
            MainApp.showSuccessScreen()
          }
        }
      case _ =>
    }
  }
}