using UnityEngine;
using System.Collections.Generic;

public class SensorPropertiesComputer {
    public SensorPropertiesComputer(
            Transform aCarTransform,
            float aMaxSensorLength,
            float aOffsetY,
            float aOffsetZ) {
        CAR_TRANSFORM = aCarTransform;
        MAX_SENSOR_LENGTH = aMaxSensorLength;
        SENSOR_OFFSET_Y = aOffsetY;
        SENSOR_OFFSET_Z = aOffsetZ;
        SENSOR_ORIGIN_BEFORE_ROTATE =
                new Vector3(
                    CAR_TRANSFORM.position.x,
                    CAR_TRANSFORM.position.y + SENSOR_OFFSET_Y,
                    CAR_TRANSFORM.position.z + SENSOR_OFFSET_Z);
    }

    public Vector3[] ComputeSensorProperties(float aCurrentSensorAngle) {
        mCarRotation = -(CAR_TRANSFORM.eulerAngles.y);
        mCarPosition = CAR_TRANSFORM.position;
        var sensorOrigin = ComputeSensorOrigin();
        var sensorDirection = ComputeSensorDirection(aCurrentSensorAngle);
        return new Vector3[]{sensorOrigin, sensorDirection};
    }

    private Vector3 ComputeSensorOrigin() {
        Vector3 sensorOrigin =
                new Vector3(
                    mCarPosition.x,
                    mCarPosition.y + SENSOR_OFFSET_Y,
                    mCarPosition.z + SENSOR_OFFSET_Z);

        sensorOrigin = ComputePointRotation(
                sensorOrigin,
                mCarPosition,
                mCarRotation);
        return sensorOrigin;
    }
    private Vector3 ComputeSensorDirection(float aCurrentSensorAngle) {
        aCurrentSensorAngle = -aCurrentSensorAngle;
        Vector3 sensorDirection = 
                new Vector3(
                    mCarPosition.x,
                    mCarPosition.y + SENSOR_OFFSET_Y,
                    mCarPosition.z + SENSOR_OFFSET_Z);
        sensorDirection.x -= MAX_SENSOR_LENGTH;
        sensorDirection = ComputePointRotation(
                sensorDirection,
                SENSOR_ORIGIN_BEFORE_ROTATE,
                aCurrentSensorAngle);
        sensorDirection = ComputePointRotation(
                sensorDirection,
                mCarPosition,
                mCarRotation);
        return sensorDirection;
    }
    private Vector3 ComputePointRotation(
            Vector3 pointToRotate,
            Vector3 rotationOrigin,
            float angleInDegrees) {
        float angleInRadians = angleInDegrees * Mathf.PI / 180.0f;
        float rotatedX =
                Mathf.Cos(angleInRadians) * (pointToRotate.x - rotationOrigin.x)
                - Mathf.Sin(angleInRadians) * (pointToRotate.z - rotationOrigin.z)
                + rotationOrigin.x;
        float rotatedZ =
                Mathf.Sin(angleInRadians) * (pointToRotate.x - rotationOrigin.x)
                + Mathf.Cos(angleInRadians) * (pointToRotate.z - rotationOrigin.z)
                + rotationOrigin.z;
        
        pointToRotate.x = rotatedX;
        pointToRotate.z = rotatedZ;
        return pointToRotate;
    }

    private readonly Vector3 SENSOR_ORIGIN_BEFORE_ROTATE;
    private readonly Transform CAR_TRANSFORM;
    private readonly float MAX_SENSOR_LENGTH;
    private readonly float SENSOR_OFFSET_Y;
    private readonly float SENSOR_OFFSET_Z;

    private float mCarRotation = 0.0f;
    private Vector3 mCarPosition;
}
