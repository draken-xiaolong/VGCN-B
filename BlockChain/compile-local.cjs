// Compile with the local solc package; no compiler-list download is required.
const fs = require('fs');
const path = require('path');
const solc = require('solc');
const sources = {};
for (const file of fs.readdirSync('contracts').filter(f => f.endsWith('.sol'))) {
  sources[`contracts/${file}`] = { content: fs.readFileSync(`contracts/${file}`, 'utf8') };
}
const output = JSON.parse(solc.compile(JSON.stringify({
  language: 'Solidity', sources,
  settings: { optimizer: { enabled: true, runs: 200 },
    outputSelection: { '*': { '*': ['abi', 'evm.bytecode', 'evm.deployedBytecode'] } } }
})));
for (const error of output.errors || []) {
  console.error(error.formattedMessage);
  if (error.severity === 'error') process.exitCode = 1;
}
if (process.exitCode) process.exit(process.exitCode);
for (const [sourceName, contracts] of Object.entries(output.contracts)) {
  for (const [contractName, c] of Object.entries(contracts)) {
    const destination = path.join('artifacts', sourceName, `${contractName}.json`);
    fs.mkdirSync(path.dirname(destination), { recursive: true });
    fs.writeFileSync(destination, JSON.stringify({
      _format: 'hh-sol-artifact-1', contractName, sourceName, abi: c.abi,
      bytecode: `0x${c.evm.bytecode.object}`,
      deployedBytecode: `0x${c.evm.deployedBytecode.object}`,
      linkReferences: c.evm.bytecode.linkReferences,
      deployedLinkReferences: c.evm.deployedBytecode.linkReferences
    }, null, 2));
  }
}
console.log(`Compiled locally with ${solc.version()}`);
