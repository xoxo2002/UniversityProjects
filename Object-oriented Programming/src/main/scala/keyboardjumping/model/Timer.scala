package keyboardjumping.model

import keyboardjumping.MainApp
import keyboardjumping.MainApp.{gc}
import scalafx.scene.paint.Color

class Timer {
  private var isTimerRunning: Boolean = false
  private var lastTime: Long = 0
  private val CountdownTime: Long = 6
  var remainingTime = CountdownTime

  def resetTimer= remainingTime = CountdownTime

  def decrementTimer(time: Long): Unit = {
    if (!isTimerRunning) {
      isTimerRunning = true
      lastTime = time
    }

    if ((time - lastTime) >= 1e9) { // 1 second has passed
      remainingTime -= 1
      lastTime = time
      gc.timeLabel.text = s"Time left: $remainingTime"
      if (remainingTime <= 3) {
        gc.timeLabel.setTextFill(Color.Red)
      }
      else {
        gc.timeLabel.setTextFill(Color.Black)
      }
      if (remainingTime <= 0) {
        MainApp.showGameOverScreen()
        remainingTime = CountdownTime
      }
    }
  }
}
