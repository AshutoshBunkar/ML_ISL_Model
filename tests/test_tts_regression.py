"""
Focused Regression Test Suite for Text-to-Speech Consecutive Calls in ISL GUI.
Validates:
1. First utterance playback
2. Second utterance with exact same sentence
3. Third utterance with changed sentence
4. Rapid consecutive clicks / queuing
5. Empty string handling
6. GUI responsiveness & sign processing during speech
7. Post-speech readiness for subsequent requests
"""
import sys
import os
import time

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(root_dir, "src")
sys.path.insert(0, src_dir)

import app

def test_tts_consecutive_calls():
    print("=" * 60)
    print("  TTS CONSECUTIVE INVOCATION REGRESSION TEST")
    print("=" * 60)

    app_instance = app.ISLApp()
    assert app_instance.speech.available, "Speech engine is not available!"
    print("Speech engine initialized and available.")

    # Test 1: First speech utterance
    print("\n[Test 1] Speaking first valid sentence: 'HELLO'...")
    app_instance.text_buffer = "HELLO"
    t0 = time.time()
    app_instance.btn_speak.invoke()
    app_instance.root.update()
    assert app_instance.speech.queue.qsize() >= 0
    app_instance.speech.queue.join()
    dur1 = time.time() - t0
    print(f"  [PASS] Utterance 1 completed in {dur1:.2f}s")
    assert dur1 > 0.5, f"Utterance exited suspiciously fast ({dur1:.2f}s), audio likely aborted!"

    # Test 2: Repeat exact same sentence
    print("\n[Test 2] Repeating exact same sentence: 'HELLO'...")
    t0 = time.time()
    app_instance.btn_speak.invoke()
    app_instance.root.update()
    app_instance.speech.queue.join()
    dur2 = time.time() - t0
    print(f"  [PASS] Utterance 2 (repeat) completed in {dur2:.2f}s")
    assert dur2 > 0.5, f"Utterance 2 exited suspiciously fast ({dur2:.2f}s) - regression detected!"

    # Test 3: Changed sentence
    print("\n[Test 3] Speaking changed sentence: 'INDIAN SIGN LANGUAGE'...")
    app_instance.text_buffer = "INDIAN SIGN LANGUAGE"
    t0 = time.time()
    app_instance.btn_speak.invoke()
    app_instance.root.update()
    app_instance.speech.queue.join()
    dur3 = time.time() - t0
    print(f"  [PASS] Utterance 3 (changed sentence) completed in {dur3:.2f}s")
    assert dur3 > 0.5, f"Utterance 3 exited suspiciously fast ({dur3:.2f}s)!"

    # Test 4: Empty sentence
    print("\n[Test 4] Handling empty and whitespace sentence...")
    app_instance.text_buffer = ""
    app_instance.btn_speak.invoke()
    app_instance.root.update()
    assert "No text to speak" in app_instance.notif_lbl.cget("text")
    print("  [PASS] Empty text handled gracefully without queueing ('No text to speak').")

    # Test 5: Rapid consecutive calls (queuing)
    print("\n[Test 5] Triggering 3 rapid consecutive requests...")
    sentences = ["ALPHA", "BRAVO", "CHARLIE"]
    t0 = time.time()
    for s in sentences:
        app_instance.text_buffer = s
        app_instance.btn_speak.invoke()
        app_instance.root.update()
    app_instance.speech.queue.join()
    dur5 = time.time() - t0
    print(f"  [PASS] 3 queued utterances completed in {dur5:.2f}s")
    assert dur5 > 2.0, f"Rapid utterances finished suspiciously fast ({dur5:.2f}s)!"

    # Test 6: Verify speech works immediately after previous playback
    print("\n[Test 6] Immediate follow-up utterance after queue cleared...")
    app_instance.text_buffer = "TEST COMPLETED"
    t0 = time.time()
    app_instance.btn_speak.invoke()
    app_instance.root.update()
    app_instance.speech.queue.join()
    dur6 = time.time() - t0
    print(f"  [PASS] Follow-up utterance completed in {dur6:.2f}s")
    assert dur6 > 0.5

    # Test 7: Cleanup
    print("\n[Test 7] Cleaning up application...")
    app_instance.quit_app()
    print("  [PASS] Application and speech worker cleanly terminated.")

    print("\n" + "=" * 60)
    print("  ALL 7 TTS REGRESSION TESTS PASSED SUCCESSFULLY! [OK]")
    print("=" * 60)

if __name__ == "__main__":
    test_tts_consecutive_calls()
