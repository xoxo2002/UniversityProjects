package keyboardjumping

import scalafx.scene.Scene
import scalafx.application.JFXApp
import scalafx.application.JFXApp.PrimaryStage
import scalafx.Includes._
import scalafxml.core.{FXMLLoader, NoDependencyResolver}
import javafx.{scene => jfxs}
import keyboardjumping.view.{GameController}


object MainApp extends JFXApp {
  //constants
  val Height = 400
  val Width = 600
  var i = 0
  var j = 0
  var m = 0
  var reachCoordinate = true


  // root component of display
  private val rootResource = getClass.getResource("view/RootLayout.fxml")
  private val rootLoader = new FXMLLoader(rootResource, NoDependencyResolver) // Load root layout from fxml file.
  rootLoader.load();
  val roots = rootLoader.getRoot[jfxs.layout.BorderPane]

  //gameDisplay component
  private val gameDisplayResources = getClass.getResource("view/GameDisplay.fxml")
  private val gameDisplayLoader = new FXMLLoader(gameDisplayResources, NoDependencyResolver)
  gameDisplayLoader.load();
  val gameDisplayRoot = gameDisplayLoader.getRoot[jfxs.layout.AnchorPane]

  // initialize stage
  stage = new PrimaryStage {
    title = "Keyboard Jumping Game"
    scene = new Scene {
      root = roots
    }
  }

  val gc = new GameController(gameDisplayRoot)

  def showGameOverScreen(): Unit = {
    show("view/GameOver.fxml")
    gc.loop.stop()
  }

  def showSuccessScreen(): Unit = {
    show("view/Success.fxml")
    gc.loop.stop()
  }

  def showMainMenu(): Unit = {
    show("view/MainMenu.fxml")
    gc.loop.stop()
    gc.timer.resetTimer
  }

  def show(resourcePath: String): Unit = {
    val resources = getClass.getResource(resourcePath)
    val loader = new FXMLLoader(resources, NoDependencyResolver)
    loader.load()
    val root = loader.getRoot[jfxs.layout.AnchorPane]
    this.roots.setCenter(root)
  }

  //start by showing main menu
  showMainMenu()
}
