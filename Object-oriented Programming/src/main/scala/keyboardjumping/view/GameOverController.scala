package keyboardjumping.view

import keyboardjumping.MainApp
import keyboardjumping.MainApp.{gc}
import scalafx.scene.control.Label
import scalafxml.core.macros.sfxml

@sfxml
class GameOverController(private val finalScoreLabel1 : Label) {
  finalScoreLabel1.text <== gc.scoreLabel.text

  def replay(): Unit = {
    MainApp.showMainMenu()
  }
}
