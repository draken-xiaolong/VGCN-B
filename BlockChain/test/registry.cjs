const { expect } = require('chai');
const { ethers } = require('hardhat');

describe('Registry security regression', function () {
  it('demonstrates cross-account replacement in the original contract', async function () {
    const [owner, attacker] = await ethers.getSigners();
    const contract = await (await ethers.getContractFactory('VectorMapRegistry')).deploy();
    await contract.connect(owner).registerVectorMap('map', 'original-cid');
    await contract.connect(attacker).registerVectorMap('map', 'attacker-cid');
    expect(await contract.getManifestCID('map')).to.equal('attacker-cid');
  });

  it('keeps claimant namespaces separate and prohibits replacing an existing record', async function () {
    const [owner, attacker] = await ethers.getSigners();
    const contract = await (await ethers.getContractFactory('AppendOnlyVectorMapRegistry')).deploy();
    const id = ethers.id('map-record');
    const digest = ethers.id('original-map');
    await contract.connect(owner).register(id, digest, 'original-cid');
    await contract.connect(attacker).register(id, digest, 'attacker-cid');
    expect((await contract.getRecord(owner.address, id)).manifestCID).to.equal('original-cid');
    expect((await contract.getRecord(attacker.address, id)).manifestCID).to.equal('attacker-cid');
    await expect(contract.connect(owner).register(id, digest, 'replacement-cid'))
      .to.be.revertedWith('record already exists');
  });
});
