package keyboardjumping.model
import scalafx.scene.image.{Image, ImageView}
import scalafx.scene.paint.Color
import scalafx.scene.shape.Rectangle

class Platform(var _x: Double, var _y: Double) {
  //setter
  def x: Double = _x
  def y: Double = _y
  val width = 160
  val height = 30
  val platformRect: Rectangle = new Rectangle {
    width = 160
    height = 30
    fill = Color.Transparent
    stroke = Color.Red
  }
  platformRect.x = _x
  platformRect.y = _y

  def getImage(): Image = {
    new Image("keyboardjumping/view/images/longpinkplatform.png")
  }

  def getView(): ImageView = {
    val platformView = new ImageView(getImage){
      //size of the picture
      fitWidth = 160
      fitHeight = 30
    }
    //assign coordinates
    platformView.layoutX = _x
    platformView.layoutY = _y
    platformView
  }

  def update(scroll: Double): Double = {
    _y += scroll
    platformRect.y = _y
    _y
  }
}
