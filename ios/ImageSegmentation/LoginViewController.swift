//
//  LoginViewController.swift
//  ImageSegmentation
//
//  Created by Yang Guo on 11/2/20.
//  Copyright © 2020 TensorFlow Authors. All rights reserved.
//

import UIKit

class LoginViewController: UIViewController {

    @IBOutlet weak var userEmailField: UITextField!
    @IBOutlet weak var userPasswordField: UITextField!
    @IBOutlet weak var loginButton: UIButton!
    override func viewDidLoad() {
        super.viewDidLoad()
        //self.view.backgroundColor = UIColor(patternImage: UIImage(named: "bg.jpg")!)
        
        let backgroundImage = UIImageView(frame: UIScreen.main.bounds)
        backgroundImage.image = UIImage(named: "bg.jpg")
        backgroundImage.contentMode =  UIView.ContentMode.scaleAspectFill
        self.view.insertSubview(backgroundImage, at: 0)
        // Do any additional setup after loading the view.
    }
    @IBAction func onSkip(_ sender: UIButton) {
        self.performSegue(withIdentifier: "loginSegue", sender: nil)
    }
    @IBAction func onLogin(_ sender: UIButton) {
        guard let email = self.userEmailField.text else {
            print("email empty")
            return
        }
        
        guard let password = self.userPasswordField.text else {
            print("email empty")
            return
        }
        //validate email
        let pattern = "[A-Z0-9a-z._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,64}"
        guard email.range(of: pattern, options:.regularExpression) != nil else {
            print("email format wrong")
            alert(message: "email format wrong")
            return
        }
        
        let host = "http://192.168.1.32:5000/mobile/login"
        let Url = String(format: host)
        guard let serviceUrl = URL(string: Url) else { return }
        let parameters: [String: Any] = [
            "request": [
                    "email": email,
                    "password": password
            ]
        ]
        var request = URLRequest(url: serviceUrl)
        request.httpMethod = "POST"
        request.setValue("Application/json", forHTTPHeaderField: "Content-Type")
        guard let httpBody = try? JSONSerialization.data(withJSONObject: parameters, options: []) else {
            return
        }
        request.httpBody = httpBody
        request.timeoutInterval = 20
        let session = URLSession.shared
        session.dataTask(with: request) { (data, response, error) in
            if let response = response {
                print(response)
            }
            if let data = data {
                do {
                    if let json = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any]{
                        
                        print(json)
                        if let loginstatus = json["status"] as? String {
                            print(loginstatus)
                            OperationQueue.main.addOperation({
                                self.performSegue(withIdentifier: "loginSegue", sender: nil)
                            })
                        
                        }
                        
                    }
                    
                    } catch {
                        print(error)
                    }
                }
            }.resume()
        
        
    }
    /*
    // MARK: - Navigation

    // In a storyboard-based application, you will often want to do a little preparation before navigation
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        // Get the new view controller using segue.destination.
        // Pass the selected object to the new view controller.
    }
    */

}


extension UIViewController {
  func alert(message: String, title: String = "") {
    let alertController = UIAlertController(title: title, message: message, preferredStyle: .alert)
    let OKAction = UIAlertAction(title: "OK", style: .default, handler: nil)
    alertController.addAction(OKAction)
    self.present(alertController, animated: true, completion: nil)
  }
}
