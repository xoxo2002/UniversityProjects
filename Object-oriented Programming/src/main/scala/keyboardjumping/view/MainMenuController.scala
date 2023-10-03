package keyboardjumping.view

import keyboardjumping.MainApp.gc
import scalafxml.core.macros.sfxml

@sfxml
class MainMenuController {
  def start(): Unit = {
    gc.start()
  }
}
