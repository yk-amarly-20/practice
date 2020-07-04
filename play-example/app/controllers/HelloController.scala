package controllers

import play.api.libs.json._
import javax.inject.Inject
import play.api.mvc._

class HelloController @Inject() (cc: ControllerComponents)
  extends AbstractController(cc) {


  def hello(): Action[AnyContent] = {
    val result: Result = Ok("Hello World")
    Action(result.as("text/plain"))
  }
 
  def helloJson(): Action[AnyContent] = Action {
    val json: JsValue = 
      Json.obj("hello" -> "world", "language" -> "scala")
    

    Ok(json)
  }
}
