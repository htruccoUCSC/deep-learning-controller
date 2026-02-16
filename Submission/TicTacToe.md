# Tic Tac Toe

## Tic Tac Toe Game

| | | |  
| | | |  
| | | |  
Turn: X  
Player X took position (1, 1).  
| | | |  
| |X| |  
| | | |  
Turn: O  
reference:  
row 0 is neutral.  
row 1 is happy.  
row 2 is surprise.  
Emotion detected as neutral (row 0). Enter 'text' to use text input instead (0, 1 or 2). Otherwise, press Enter to continue.

reference:  
col 0 is neutral.  
col 1 is happy.  
col 2 is surprise.  
Emotion detected as neutral (col 0). Enter 'text' to use text input instead (0, 1 or 2). Otherwise, press Enter to continue.

Player O took position (0, 0).  
|O| | |  
| |X| |  
| | | |  
Turn: X  
Player X took position (2, 2).  
|O| | |  
| |X| |  
| | |X|  
Turn: O  
reference:  
row 0 is neutral.  
row 1 is happy.  
row 2 is surprise.  
Emotion detected as happy (row 1). Enter 'text' to use text input instead (0, 1 or 2). Otherwise, press Enter to continue.  

reference:  
col 0 is neutral.  
col 1 is happy.  
col 2 is surprise.  
Emotion detected as neutral (col 0). Enter 'text' to use text input instead (0, 1 or 2). Otherwise, press Enter to continue.  

Player O took position (1, 0).  
|O| | |  
|O|X| |  
| | |X|  
Turn: X  
Player X took position (0, 1).  
|O|X| |  
|O|X| |  
| | |X|  
Turn: O  
reference:  
row 0 is neutral.  
row 1 is happy.  
row 2 is surprise.  
Emotion detected as surprise (row 2). Enter 'text' to use text input instead (0, 1 or 2). Otherwise, press Enter to continue.

reference:  
col 0 is neutral.  
col 1 is happy.  
col 2 is surprise.  
Emotion detected as neutral (col 0). Enter 'text' to use text input instead (0, 1 or 2). Otherwise, press Enter to continue.  

Player O took position (2, 0).  
|O|X| |  
|O|X| |  
|O| |X|  
Player O has won!  

## Questions

- How well did your interface work?
  - It worked well as long as I framed my face to take up about 80% of the shot.
- Did it recognize your facial expressions with the same accuracy as it achieved against the test set?
  - It detected facial expressions pretty well about as well as you would expect for a 75% accurate model. It struggled with suprise a bit. I was able to get through the game without having to overide any of my choices though.

## Code

```(Python)
    def _get_emotion(self, img) -> int:
        if not hasattr(self, "_model"):
            self._model = models.load_model("results/basic_model_40_epochs_timestamp_1771195657.keras")

        resized = cv2.resize(img, image_size)
        resized = resized.astype("float32")

        if len(resized.shape) == 2:
            resized = np.stack([resized, resized, resized], axis=-1)
        resized = np.expand_dims(resized, axis=0)

        predictions = self._model.predict(resized, verbose=0)
        emotion = int(np.argmax(predictions[0]))
        return emotion
```
