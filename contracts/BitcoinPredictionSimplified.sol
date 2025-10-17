// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title BitcoinPredictionSimplified
 * @dev Minimal smart contract for storing immutable Bitcoin price predictions
 *
 * ONLY stores predicted prices (1d, 3d, 7d) - no current or actual prices needed
 * All other data (current prices, actual prices, metrics) handled off-chain
 *
 * Blockchain proves: WHAT was predicted and WHEN
 * Off-chain proves: Current/actual prices (publicly verifiable from Bitcoin APIs)
 *
 * Deployed on Moonbase Alpha testnet
 *
 * Author: Bitcoin Price Prediction System
 * Date: October 2025
 */
contract BitcoinPredictionSimplified {

    // ========================================================================
    // STRUCTS
    // ========================================================================

    struct Prediction {
        uint256 timestamp;          // Unix timestamp when prediction was made (from block.timestamp)
        uint256 predictedPrice1d;   // Predicted price 1 day later (in cents)
        uint256 predictedPrice3d;   // Predicted price 3 days later (in cents)
        uint256 predictedPrice7d;   // Predicted price 7 days later (in cents)
        bytes32 modelHash;          // Hash of model version (e.g., "xgboost_v1")
    }

    // ========================================================================
    // STATE VARIABLES
    // ========================================================================

    Prediction[] public predictions;
    address public owner;
    uint256 public totalPredictions;

    // ========================================================================
    // EVENTS
    // ========================================================================

    event PredictionStored(
        uint256 indexed predictionId,
        uint256 timestamp,
        uint256 predictedPrice1d,
        uint256 predictedPrice3d,
        uint256 predictedPrice7d,
        bytes32 modelHash
    );

    // ========================================================================
    // MODIFIERS
    // ========================================================================

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }

    // ========================================================================
    // CONSTRUCTOR
    // ========================================================================

    constructor() {
        owner = msg.sender;
        totalPredictions = 0;
    }

    // ========================================================================
    // MAIN FUNCTIONS
    // ========================================================================

    /**
     * @dev Store a new prediction (all 3 horizons in one transaction)
     * @param _predictedPrice1d Predicted price 1 day later in cents
     * @param _predictedPrice3d Predicted price 3 days later in cents
     * @param _predictedPrice7d Predicted price 7 days later in cents
     * @param _modelHash Hash of the model version used
     * @return predictionId The ID of the stored prediction
     */
    function storePrediction(
        uint256 _predictedPrice1d,
        uint256 _predictedPrice3d,
        uint256 _predictedPrice7d,
        bytes32 _modelHash
    ) external onlyOwner returns (uint256) {

        // Validate inputs
        require(_predictedPrice1d > 0, "Predicted price 1d must be > 0");
        require(_predictedPrice3d > 0, "Predicted price 3d must be > 0");
        require(_predictedPrice7d > 0, "Predicted price 7d must be > 0");

        // Create new prediction
        Prediction memory newPrediction = Prediction({
            timestamp: block.timestamp,
            predictedPrice1d: _predictedPrice1d,
            predictedPrice3d: _predictedPrice3d,
            predictedPrice7d: _predictedPrice7d,
            modelHash: _modelHash
        });

        // Store prediction
        predictions.push(newPrediction);
        totalPredictions++;

        uint256 predictionId = predictions.length - 1;

        emit PredictionStored(
            predictionId,
            block.timestamp,
            _predictedPrice1d,
            _predictedPrice3d,
            _predictedPrice7d,
            _modelHash
        );

        return predictionId;
    }

    // ========================================================================
    // VIEW FUNCTIONS
    // ========================================================================

    /**
     * @dev Get a single prediction by ID
     * @param _predictionId ID of the prediction
     * @return Prediction struct
     */
    function getPrediction(uint256 _predictionId) external view returns (Prediction memory) {
        require(_predictionId < predictions.length, "Invalid prediction ID");
        return predictions[_predictionId];
    }

    /**
     * @dev Get multiple recent predictions
     * @param _count Number of recent predictions to fetch
     * @return Array of Prediction structs
     */
    function getRecentPredictions(uint256 _count) external view returns (Prediction[] memory) {
        uint256 count = _count > predictions.length ? predictions.length : _count;
        Prediction[] memory recent = new Prediction[](count);

        for (uint256 i = 0; i < count; i++) {
            recent[i] = predictions[predictions.length - count + i];
        }

        return recent;
    }

    /**
     * @dev Get all predictions (use with caution - can be gas intensive for many predictions)
     * @return Array of all Prediction structs
     */
    function getAllPredictions() external view returns (Prediction[] memory) {
        return predictions;
    }

    /**
     * @dev Get total number of predictions
     * @return Total number of predictions stored
     */
    function getPredictionCount() external view returns (uint256) {
        return predictions.length;
    }

    /**
     * @dev Get prediction details by ID (expanded view)
     * @param _predictionId ID of the prediction
     * @return timestamp When prediction was made
     * @return predictedPrice1d Predicted price for 1 day
     * @return predictedPrice3d Predicted price for 3 days
     * @return predictedPrice7d Predicted price for 7 days
     * @return modelHash Model version hash
     */
    function getPredictionDetails(uint256 _predictionId) external view returns (
        uint256 timestamp,
        uint256 predictedPrice1d,
        uint256 predictedPrice3d,
        uint256 predictedPrice7d,
        bytes32 modelHash
    ) {
        require(_predictionId < predictions.length, "Invalid prediction ID");
        Prediction memory pred = predictions[_predictionId];
        return (pred.timestamp, pred.predictedPrice1d, pred.predictedPrice3d, pred.predictedPrice7d, pred.modelHash);
    }

    // ========================================================================
    // ADMIN FUNCTIONS
    // ========================================================================

    /**
     * @dev Transfer ownership to a new address
     * @param _newOwner Address of the new owner
     */
    function transferOwnership(address _newOwner) external onlyOwner {
        require(_newOwner != address(0), "New owner cannot be zero address");
        owner = _newOwner;
    }

    /**
     * @dev Get contract metadata
     * @return _owner Contract owner address
     * @return _totalPredictions Total predictions stored
     * @return _contractBalance Contract balance in wei
     */
    function getContractInfo() external view returns (
        address _owner,
        uint256 _totalPredictions,
        uint256 _contractBalance
    ) {
        return (owner, totalPredictions, address(this).balance);
    }
}
