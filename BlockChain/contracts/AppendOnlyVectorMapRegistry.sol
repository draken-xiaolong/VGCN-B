// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @notice Revision prototype. Registration is evidence of a wallet's claim,
/// not proof that the wallet owns copyright in the referenced map.
contract AppendOnlyVectorMapRegistry {
    struct Record {
        string manifestCID;
        bytes32 contentDigest;
        uint256 timestamp;
    }
    mapping(address => mapping(bytes32 => Record)) private records;
    event Registered(address indexed claimant, bytes32 indexed recordId,
                     bytes32 contentDigest, string manifestCID, uint256 timestamp);

    function register(bytes32 recordId, bytes32 contentDigest, string calldata cid) external {
        require(recordId != bytes32(0), "empty record id");
        require(contentDigest != bytes32(0), "empty digest");
        require(bytes(cid).length != 0, "empty cid");
        require(bytes(records[msg.sender][recordId].manifestCID).length == 0,
                "record already exists");
        records[msg.sender][recordId] = Record(cid, contentDigest, block.timestamp);
        emit Registered(msg.sender, recordId, contentDigest, cid, block.timestamp);
    }

    function getRecord(address claimant, bytes32 recordId) external view returns (Record memory) {
        return records[claimant][recordId];
    }
}
