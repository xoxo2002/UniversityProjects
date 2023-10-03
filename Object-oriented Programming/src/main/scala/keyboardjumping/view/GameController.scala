package keyboardjumping.view

import scalafx.animation.AnimationTimer
import scalafx.scene.image.{Image, ImageView}
import scalafx.scene.layout.HBox
import scalafx.Includes._
import javafx.{scene => jfxs}
import keyboardjumping.MainApp
import keyboardjumping.controller.InputController
import keyboardjumping.model.{LivesBar, Platform, Player, Timer}
import scala.collection.mutable.ListBuffer
import keyboardjumping.MainApp.{Width, gc, reachCoordinate, i, j, m}
import scalafx.scene.control.Label
import scalafx.scene.input.{KeyEvent}
import scalafx.scene.shape.Line

class GameController (gameDisplayRoot: jfxs.layout.AnchorPane){
  var platforms: ListBuffer[Platform] = ListBuffer.empty
  val platformsImageView: ListBuffer[ImageView] = ListBuffer.empty
  val words: ListBuffer[String] = ListBuffer.empty
  val wordsView: ListBuffer[HBox] = ListBuffer.empty
  val startingPositionX = 406
  val startingPositionY = 300
  val image = new Image("keyboardjumping/view/images/facingfront.png")
  val player = new Player(startingPositionX, startingPositionY)
  val playerImageView = new ImageView(image) {
    fitWidth = player.playerWidth
    fitHeight = player.playerHeight
  }
  gameDisplayRoot.children.add(playerImageView)
  var isJumping = false
  playerImageView.layoutX = player.x
  playerImageView.layoutY = player.y

  var totalScoreSoFar = 0
  var totalFault = 0
  var startTime: Long = 0
  var elapsedTime: Long = 0
  var q = 0

  private val platformController = new PlatformController(gameDisplayRoot, platforms, platformsImageView, wordsView) //platformController
  private val inputController = new InputController(wordsView) //inputController
  private val playerController = new PlayerController(player, playerImageView, platforms)

  val ScrollThreshold = 30
  var scroll = 0.0

  val timeLabel = new Label {
    text = "Time you have left: 0"
    style = "-fx-font-size: 18px;" // Optional: Set the font size
    layoutX = 10
    layoutY = 65
  }
  gameDisplayRoot.children.add(timeLabel)

  //scorelabel
  val scoreLabel = new Label {
    text = "Score: 0"
    style = "-fx-font-size: 18px;" // Optional: Set the font size
    layoutX = 10
    layoutY = 40
  }
  gameDisplayRoot.children.add(scoreLabel)
  gameDisplayRoot.children.add(LivesBar.livesIndicator)

  //threshold line, keeps player in the screen
  val line = new Line {
    startX = 0
    startY = ScrollThreshold
    endX = Width
    endY = ScrollThreshold
  }


  val timer = new Timer()
  //loop for game display
  val loop = AnimationTimer { time =>
    //check for key events
    timer.decrementTimer(time)
    playerController.handleMovement()
    playerController.checkPlatformCollision()

    //Update platforms while scrolling
    platformController.updatePlatformsPosition()
    playerController.handleThresholdCollision()
    playerController.updatePlayerImage()
  }

    def start(): Unit = {
      platformController.currentPlatformX = 130
      platformController.currentPlatformY = 270
      q = 0
      scroll = 0.0
      totalScoreSoFar = 0
      totalFault = 0
      startTime = 0
      isJumping = false
      scoreLabel.text = s"Score: $totalScoreSoFar"

      //keeping track of words, wordsImageView, platforms, platforms image view
      reachCoordinate = true
      i = 0
      j = 0
      m = 0

      //println(s"Children count before removal: ${gameDisplayRoot.children.size}")
      wordsView.foreach(gameDisplayRoot.children.remove)
      platformsImageView.foreach(gameDisplayRoot.children.remove)
      platforms.foreach(gameDisplayRoot.children.remove)
      //println(s"Children count after removal: ${gameDisplayRoot.children.size}")

      // Clear platforms and words
      platforms.clear()
      platformsImageView.clear()
      words.clear()
      wordsView.clear()
      print(platforms.length, platformsImageView.length, words.length, wordsView.length)

      // Reset player position and appearance
      player.x = startingPositionX
      player.y = startingPositionY
      playerImageView.layoutX = player.x
      playerImageView.layoutY = player.y
      platformController.currentPlatformX = 130
      platformController.currentPlatformY = 270

      //reset lives indicator
      LivesBar.resetLives()

      //generate new platforms and start the game loop
      platformController.generatePlatform()

      //handle keyboard input events
      gameDisplayRoot.onKeyPressed = (event: KeyEvent) => {
        inputController.onKeyPressed(event)
      }
      gameDisplayRoot.onKeyReleased = (event: KeyEvent) => {
        inputController.onKeyReleased(event)
      }
      MainApp.roots.setCenter(gameDisplayRoot)
      gc.loop.start()
    }

}
