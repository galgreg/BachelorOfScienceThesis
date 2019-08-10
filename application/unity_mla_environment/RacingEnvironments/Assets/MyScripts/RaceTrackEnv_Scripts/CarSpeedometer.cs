using UnityEngine;

public class CarSpeedometer
{
    public float GetNormalizedVelocity() {
        float carVelocity = mCarRigidbody.velocity.magnitude;
        return carVelocity / CAR_MAX_VELOCITY;
    }
    public Rigidbody mCarRigidbody;
    public float CAR_MAX_VELOCITY = 0.78f;
}
