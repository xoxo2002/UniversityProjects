package keyboardjumping.view

import keyboardjumping.MainApp
import scalafxml.core.macros.sfxml

  @sfxml
class GameDisplayController() {
  def main(): Unit = {
    MainApp.showMainMenu()
  }
}
